"""
Parser debugger for language processing.

This module provides debugging capabilities for the language parser,
allowing step-by-step inspection of the parsing process.
"""

import json
from collections import defaultdict
from .language_areas import *

class ParserDebugger:
    """Debugger for step-by-step parser inspection."""
    
    def __init__(self, brain, all_areas, explicit_areas):
        """
        Initialize the parser debugger.
        
        Args:
            brain: The parser brain instance
            all_areas: List of all language areas
            explicit_areas: List of explicit language areas
        """
        self.b = brain
        self.all_areas = all_areas
        self.explicit_areas = explicit_areas

    def run(self):
        """Run the interactive debugger."""
        command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
        while command:
            if command == "P":
                self.peak()
                return
            elif command:
                print("DEBUGGER: Command not recognized...")
                command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
            else:
                return

    def peak(self):
        """Peak at the current state of the parser."""
        remove_map = defaultdict(int)
        # Temporarily set beta to 0
        self.b.disable_plasticity = True
        self.b.save_winners = True

        for area in self.all_areas:
            self.b.area_by_name[area].unfix_assembly()
            
        while True:
            test_proj_map_string = input("DEBUGGER: enter projection map, eg. {\"VERB\": [\"LEX\"]}, or ENTER to quit\n")
            if not test_proj_map_string:
                break
                
            test_proj_map = json.loads(test_proj_map_string)
            # Important: save winners to later "remove" this test project round 
            to_area_set = set()
            for _, to_area_list in test_proj_map.items():
                for to_area in to_area_list:
                    to_area_set.add(to_area)
                    if not self.b.area_by_name[to_area].saved_winners:
                        self.b.area_by_name[to_area].saved_winners.append(self.b.area_by_name[to_area].winners)

            for to_area in to_area_set:
                remove_map[to_area] += 1

            self.b.project({}, test_proj_map)
            for area in self.explicit_areas:
                if area in to_area_set:
                    area_word = self.b.interpretAssemblyAsString(area)
                    print("DEBUGGER: in explicit area " + area + ", got: " + area_word)

            print_assemblies = input("DEBUGGER: print assemblies in areas? Eg. 'LEX,VERB' or ENTER to cont\n")
            if not print_assemblies:
                continue
            for print_area in print_assemblies.split(","):
                print("DEBUGGER: Printing assembly in area " + print_area)
                print(str(self.b.area_by_name[print_area].winners))
                if print_area in self.explicit_areas:
                    word = self.b.interpretAssemblyAsString(print_area)
                    print("DEBUGGER: in explicit area got assembly = " + word)

        # Restore assemblies (winners) and w values to before test projections
        for area, num_test_projects in remove_map.items():
            self.b.area_by_name[area].winners = self.b.area_by_name[area].saved_winners[0]
            self.b.area_by_name[area].w = self.b.area_by_name[area].saved_w[-num_test_projects - 1]
            self.b.area_by_name[area].saved_w = self.b.area_by_name[area].saved_w[:(-num_test_projects)]
        self.b.disable_plasticity = False
        self.b.save_winners = False
        for area in self.all_areas:
            self.b.area_by_name[area].saved_winners = []
