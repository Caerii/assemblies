//! Specification: neural_assemblies/ir/VERIFICATION.md#contract-competition-wire
//! Configuration transport, not a Rust selection backend.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

static VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| super::schema_validator(include_str!("../v1/competition.schema.json")));

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct CompetitionDocument(Value);

// Counts use exact JSON integer spellings, so bounds remain distinguishable
// beyond binary64 and u64. JSON syntax already excludes leading zeros.
fn count(value: &Value, name: &str) -> Result<Option<String>, String> {
    if value[name].is_null() {
        return Ok(None);
    }
    let spelling = value[name].to_string();
    if !spelling.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{name} must be a JSON integer"));
    }
    Ok(Some(spelling))
}

impl TryFrom<Value> for CompetitionDocument {
    type Error = String;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        VALIDATOR.validate(&value).map_err(|e| e.to_string())?;
        count(&value, "k")?;
        let minimum = count(&value, "min_winners")?;
        let maximum = count(&value, "max_winners")?;
        if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
            if (minimum.len(), &minimum) > (maximum.len(), &maximum) {
                return Err("max_winners must be >= min_winners".into());
            }
        }
        for name in ["threshold", "fraction_of_max", "e_fraction", "sigma_c"] {
            if !value[name].is_null() && !value[name].as_f64().is_some_and(f64::is_finite) {
                return Err(format!("{name} must fit a finite float"));
            }
        }
        Ok(Self(value))
    }
}

impl CompetitionDocument {
    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_competition_corpus() {
        let cases: Value =
            serde_json::from_str(include_str!("../v1/competition.cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let parsed = serde_json::from_value::<CompetitionDocument>(case["document"].clone());
            assert_eq!(
                parsed.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}: {:?}",
                case["name"],
                parsed
            );
            if let Ok(document) = parsed {
                assert_eq!(document.as_value(), &case["document"]);
                assert_eq!(serde_json::to_value(&document).unwrap(), case["document"]);
            }
        }
    }
}
