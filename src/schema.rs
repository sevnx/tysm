use schemars::transform::{transform_subschemas, Transform};
use schemars::Schema;
use serde_json::Value;

pub struct OpenAiTransform;

impl Transform for OpenAiTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            if obj.get("$ref").is_none() {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
                obj.remove("format");
                obj.remove("minimum");
                obj.remove("maximum");

                // get all the items under "properties"
                let properties = obj
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(|obj| {
                        obj.keys()
                            .map(|k| Value::String(k.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                // add all the properties to the "required" array
                obj.insert("required".to_string(), Value::Array(properties));
            }
        }
        transform_subschemas(self, schema);
    }
}
