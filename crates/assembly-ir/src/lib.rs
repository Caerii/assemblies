//! Assembly Calculus IR v1 — protocol documents for cross-language parity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const IR_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolDocument {
    pub ir_version: String,
    pub protocol: String,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub metrics: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub regimes: Option<HashMap<String, HashMap<String, serde_json::Value>>>,
    #[serde(default)]
    pub thresholds: Option<HashMap<String, serde_json::Value>>,
}

impl ProtocolDocument {
    pub fn validate(&self) -> Result<(), String> {
        if self.ir_version != IR_VERSION {
            return Err(format!(
                "ir_version must be {IR_VERSION}, got {}",
                self.ir_version
            ));
        }
        if self.protocol.is_empty() {
            return Err("protocol id must be non-empty".into());
        }
        Ok(())
    }
}

pub fn parse_protocol_json(s: &str) -> Result<ProtocolDocument, serde_json::Error> {
    serde_json::from_str(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cross_lang_golden_shape() {
        let sample = r#"{
            "ir_version": "1",
            "protocol": "cross_lang.pnas_scaling",
            "backend": "python",
            "metrics": {},
            "regimes": {
                "ci_parity": {
                    "chance_overlap": 0.016,
                    "project_persistence": 1.0,
                    "separate_overlap": 0.025
                }
            }
        }"#;
        let doc = parse_protocol_json(sample).unwrap();
        doc.validate().unwrap();
        assert_eq!(doc.protocol, "cross_lang.pnas_scaling");
    }
}
