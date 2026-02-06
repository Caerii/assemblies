"""Debug the verb pathway issue"""
import torch
import numpy as np

DEVICE = torch.device('cuda')

from nemo_gpu_corrected import NemoLanguageSystemGPU

system = NemoLanguageSystemGPU(n_neurons=5000, cap_size=50, density=0.1)

# Create a few words
nouns = ['dog', 'cat']
verbs = ['runs', 'walks']

for word in nouns:
    system.create_assembly(word, 'Phon')
    system.create_assembly(word, 'Visual')
for word in verbs:
    system.create_assembly(word, 'Phon')
    system.create_assembly(word, 'Motor')

# Train
for _ in range(50):
    for noun in nouns:
        for verb in verbs:
            system.present_grounded_sentence(noun, verb)

# Debug: check Lex2 -> Motor weights
print('Weight statistics:')
print(f'  Motor→Lex2 mean: {system.W_motor_lex2[system.W_motor_lex2 > 0].mean():.2f}')
print(f'  Motor→Lex2 max: {system.W_motor_lex2.max():.2f}')
print(f'  Lex2→Motor mean: {system.W_lex2_motor[system.W_lex2_motor > 0].mean():.2f}')
print(f'  Lex2→Motor max: {system.W_lex2_motor.max():.2f}')

# Test pathway for 'runs'
word = 'runs'
phon = system.phon_assemblies[word]
motor = system.motor_assemblies[word]

print(f'\nPathway test for "{word}":')
print(f'  Phon assembly: first 5 indices = {phon[:5].cpu().numpy()}')
print(f'  Motor assembly: first 5 indices = {motor[:5].cpu().numpy()}')

# Fire Phon -> Lex2
system.lex2.inhibit()
for _ in range(3):
    total_input = system.lex2.W_inp[phon].sum(dim=0)
    if system.lex2.activated is not None:
        total_input += system.lex2.W_rec[system.lex2.activated].sum(dim=0)
    _, winners = torch.topk(total_input, system.k)
    system.lex2.activated = winners

print(f'  Lex2 activation: first 5 indices = {system.lex2.activated[:5].cpu().numpy()}')

# Check Lex2 -> Motor weights for these neurons
lex2_active = system.lex2.activated
motor_input = system.W_lex2_motor[lex2_active].sum(dim=0)
print(f'  Motor input from Lex2: max = {motor_input.max():.2f}, sum = {motor_input.sum():.2f}')

_, top_motor = torch.topk(motor_input, system.k)
motor_set = set(motor.cpu().numpy())
top_set = set(top_motor.cpu().numpy())
overlap = len(motor_set & top_set) / system.k
print(f'  Top motor neurons overlap with target: {overlap:.1%}')

# Compare to noun pathway
word = 'dog'
phon = system.phon_assemblies[word]
visual = system.visual_assemblies[word]

system.lex1.inhibit()
for _ in range(3):
    total_input = system.lex1.W_inp[phon].sum(dim=0)
    if system.lex1.activated is not None:
        total_input += system.lex1.W_rec[system.lex1.activated].sum(dim=0)
    _, winners = torch.topk(total_input, system.k)
    system.lex1.activated = winners

visual_input = system.W_lex1_visual[system.lex1.activated].sum(dim=0)
_, top_visual = torch.topk(visual_input, system.k)
visual_set = set(visual.cpu().numpy())
top_set = set(top_visual.cpu().numpy())
overlap = len(visual_set & top_set) / system.k
print(f'\nNoun pathway (dog) overlap: {overlap:.1%}')
print(f'  Visual input from Lex1: max = {visual_input.max():.2f}, sum = {visual_input.sum():.2f}')

