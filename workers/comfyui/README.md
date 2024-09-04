This is the base PyWorker for comfyui. It can be used to create PyWorker that use various models and
workflows. It provides two endpoints:

1. `/prompt`: Uses the default comfy workflow defined under `misc/default_workflows`
2. `/custom_workflow`: Allows the client to send their own comfy workflow with each API request.

To use the comfyui PyWorker, `$COMFY_MODEL` env variable must be set in the template. Current options are
`sd3` and `flux`. Each have example clients.

To add new models, a JSON with name `$COMFY_MODEL.json` must be created under `misc/default_workflows`

NOTE: default workflows follow this format:

```json
{
  "input": {
    "handler": "RawWorkflow",
    "aws_access_key_id": "your-s3-access-key",
    "aws_secret_access_key": "your-s3-secret-access-key",
    "aws_endpoint_url": "https://my-endpoint.backblaze.com",
    "aws_bucket_name": "your-bucket",
    "webhook_url": "your-webhook-url",
    "webhook_extra_params": {},
    "workflow_json": {}
  }
}
```

You can ignore all of these fields except for `workflow_json`.

Fields written as "{{FOO}}" will be replaced using data from a user request. For example, SD3's workflow has the
following nodes:

```json
      "5": {
        "inputs": {
          "width": "{{WIDTH}}",
          "height": "{{HEIGHT}}",
          "batch_size": 1
        },

      "6": {
        "inputs": {
          "text": "{{PROMPT}}",
          "clip": ["11", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      ...
      "17": {
        "inputs": {
          "scheduler": "simple",
          "steps": "{{STEPS}}",
          "denoise": 1,
          "model": ["12", 0]
        },
        "class_type": "BasicScheduler",
        "_meta": {
          "title": "BasicScheduler"
        }
      },
      ...
      "25": {
        "inputs": {
          "noise_seed": "{{SEED}}"
        },
        "class_type": "RandomNoise",
        "_meta": {
          "title": "RandomNoise"
        }
      }

```

Incoming requests have the following JSON format:

```json
{
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
}
```

Each value in those fields with replace the placeholder of the same name in the default workflow.

See Vast's serverless documentation for more details on how to use comfyui with autoscaler
