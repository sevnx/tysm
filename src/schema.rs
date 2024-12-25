use schemars::transform::{transform_subschemas, Transform};
use schemars::{json_schema, Schema};
use serde_json::{json, Map, Value};

pub struct OpenAiTransform;

impl Transform for OpenAiTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            if let None = obj.get("$ref") {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }
        }
        transform_subschemas(self, schema);
    }
}
