use schemars::transform::{transform_subschemas, Transform};
use schemars::Schema;
use serde_json::Value;

pub struct OpenAiTransform;

impl Transform for OpenAiTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            if obj.get("$ref").is_none() {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }
        }
        transform_subschemas(self, schema);
    }
}
