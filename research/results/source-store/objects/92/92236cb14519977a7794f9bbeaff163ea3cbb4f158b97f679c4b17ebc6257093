//! Specification: neural_assemblies/ir/VERIFICATION.md#contract-homeostasis-wire
//! Configuration transport only; this is not a Rust execution backend.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

static VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| super::schema_validator(include_str!("../v1/homeostasis.schema.json")));

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct HomeostasisDocument(Value);

impl TryFrom<Value> for HomeostasisDocument {
    type Error = String;
    fn try_from(mut value: Value) -> Result<Self, Self::Error> {
        VALIDATOR
            .validate(&value)
            .map_err(|error| error.to_string())?;
        if let Some(scope) = value["synaptic_scaling"].as_array_mut() {
            scope.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
        }
        Ok(Self(value))
    }
}

impl HomeostasisDocument {
    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_homeostasis_wire() {
        let cases: Value =
            serde_json::from_str(include_str!("../v1/homeostasis.cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let parsed = serde_json::from_value::<HomeostasisDocument>(case["document"].clone());
            assert_eq!(
                parsed.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
            if let Ok(document) = parsed {
                let mut expected = case["document"].clone();
                if let Some(scope) = expected["synaptic_scaling"].as_array_mut() {
                    scope.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                }
                assert_eq!(serde_json::to_value(&document).unwrap(), expected);
                assert_eq!(document.as_value(), &expected);
            }
        }
    }
}
