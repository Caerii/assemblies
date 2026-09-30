//! Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-round
//! Wire validation only; numerical execution remains a separate backend obligation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

static VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| super::schema_validator(include_str!("../v1/explicit-round.schema.json")));

/// A lossless explicit-round document whose constructor enforces the shared schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct ExplicitRoundDocument(Value);

impl TryFrom<Value> for ExplicitRoundDocument {
    type Error = String;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if !super::finite_numbers(&value) {
            return Err("decimal/exponent numbers must fit finite binary64".into());
        }
        VALIDATOR
            .validate(&value)
            .map_err(|error| error.to_string())?;
        Ok(Self(value))
    }
}

impl ExplicitRoundDocument {
    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

pub fn parse_explicit_round_json(s: &str) -> Result<ExplicitRoundDocument, serde_json::Error> {
    serde_json::from_str(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_explicit_round_wire() {
        let cases: Value =
            serde_json::from_str(include_str!("../v1/explicit-round.cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let encoded = case["document"].to_string();
            let wire = case["raw_json"].as_str().unwrap_or(&encoded);
            let parsed = parse_explicit_round_json(wire);
            assert_eq!(
                parsed.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
            if let Ok(document) = parsed {
                assert_eq!(document.as_value(), &case["document"], "{}", case["name"]);
            }
        }
    }
}
