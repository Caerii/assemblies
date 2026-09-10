//! Protocol IR is an evidence document, not an executable projection program.
//! Schema: neural_assemblies/ir/v1/protocol.schema.json
//! Specification: neural_assemblies/ir/VERIFICATION.md#contract-protocol-wire

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

pub mod competition;
pub mod homeostasis;

pub const IR_VERSION: &str = "1";
fn schema_validator(source: &str) -> jsonschema::Validator {
    let schema: Value = serde_json::from_str(source).expect("packaged schema is valid JSON");
    jsonschema::draft202012::options()
        .should_validate_formats(false)
        .build(&schema)
        .expect("packaged schema is valid")
}

static VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| schema_validator(include_str!("../v1/protocol.schema.json")));

/// Validated, lossless protocol document; construction cannot bypass validation.
/// Retaining the entire object preserves extension fields and omitted properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct ProtocolDocument(Value);

// Match the Python boundary: decimal/exponent numbers must fit finite binary64;
// integer identities remain arbitrary precision. This is not a numeric proof.
fn finite_numbers(value: &Value) -> bool {
    match value {
        Value::Number(number) => {
            let spelling = number.to_string();
            !spelling.contains(['.', 'e', 'E']) || number.as_f64().is_some_and(f64::is_finite)
        }
        Value::Array(values) => values.iter().all(finite_numbers),
        Value::Object(values) => values.values().all(finite_numbers),
        _ => true,
    }
}

impl TryFrom<Value> for ProtocolDocument {
    type Error = String;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if !finite_numbers(&value) {
            return Err("decimal/exponent numbers must fit finite binary64".into());
        }
        VALIDATOR
            .validate(&value)
            .map_err(|error| error.to_string())?;
        Ok(Self(value))
    }
}

impl From<ProtocolDocument> for Value {
    fn from(document: ProtocolDocument) -> Self {
        document.0
    }
}

impl ProtocolDocument {
    pub fn as_value(&self) -> &Value {
        &self.0
    }
    pub fn protocol(&self) -> &str {
        self.0["protocol"]
            .as_str()
            .expect("validated protocol string")
    }
    pub fn validate(&self) -> Result<(), String> {
        VALIDATOR
            .validate(&self.0)
            .map_err(|error| error.to_string())
    }
}

pub fn parse_protocol_json(s: &str) -> Result<ProtocolDocument, serde_json::Error> {
    serde_json::from_str(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_wire_contract() {
        let cases: Value = serde_json::from_str(include_str!("../v1/protocol.cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let input = &case["document"];
            let encoded = input.to_string();
            let wire = case["raw_json"].as_str().unwrap_or(&encoded);
            let parsed = parse_protocol_json(wire);
            assert_eq!(
                parsed.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
            if let Ok(document) = parsed {
                assert_eq!(
                    serde_json::to_value(document).unwrap(),
                    *input,
                    "{}",
                    case["name"]
                );
            }
        }
    }
}
