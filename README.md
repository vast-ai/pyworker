# Vast PyWorker

Vast PyWorker is a Python web server designed to run alongside a LLM or image generation models running on vast,
enabling autoscaler integration.
It serves as the primary entry point for API requests, forwarding them to the model's API hosted on the
same instance. Additionally, it monitors performance metrics and estimates current workload based on factors
such as the number of tokens processed for LLMs or image resolution and steps for image generation models,
reporting these metrics to the autoscaler.

## How to Use

If you want to use autoscaler, you just need to use one of Vast's autoscaler templates. If you'd like to
implement PyWorker for a template that is not marked as autoscaler compatible on Vast, refer to
`workers/hello_world/README.md`
