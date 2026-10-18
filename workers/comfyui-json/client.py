import os
import sys
import json
import uuid
import random
import asyncio
import logging
import argparse
import aiohttp

from vastai import Serverless

# ---------------------- Config ----------------------
DEFAULT_PROMPT = "a beautiful sunset over mountains, digital art, highly detailed"
ENDPOINT_NAME = "my-comfyui-endpoint"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 20
COST = 100  # Fixed cost for ComfyUI requests

# Optional S3 Configuration (from environment variables)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_s3_client():
    """Create and return an S3 client configured for the S3-compatible endpoint"""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log.error("boto3 is required for S3 uploads. Install with: pip install boto3")
        return None

    if not all([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY]):
        log.error("S3 environment variables not fully configured. Required:")
        log.error("  S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY")
        return None

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


# ---------------------- API Functions ----------------------
async def call_generate(
    client: Serverless,
    *,
    endpoint_name: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict:
    """Generate image using Text2Image modifier"""
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "modifier": "Text2Image",
            "modifications": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
        }
    }
    return await endpoint.request("/generate/sync", payload, cost=COST)


async def call_generate_workflow(
    client: Serverless,
    *,
    endpoint_name: str,
    workflow_json: dict,
) -> dict:
    """Generate using custom workflow JSON"""
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "workflow_json": workflow_json,
        }
    }
    return await endpoint.request("/generate/sync", payload, cost=COST)


# ---------------------- Demo Class ----------------------
class APIDemo:
    def __init__(self, client: Serverless, endpoint_name: str, upload_s3: bool = False):
        self.client = client
        self.endpoint_name = endpoint_name
        self.upload_s3 = upload_s3
        self.s3_client = get_s3_client() if upload_s3 else None
        
        if upload_s3 and not self.s3_client:
            log.warning("S3 upload requested but client creation failed. Images will only be saved locally.")

    def extract_filename(self, response: dict) -> str | None:
        """Extract the generated image filename from ComfyUI response"""
        if "comfyui_response" in response:
            for data in response["comfyui_response"].values():
                if isinstance(data, dict) and "outputs" in data:
                    for node_output in data["outputs"].values():
                        if "images" in node_output and node_output["images"]:
                            return node_output["images"][0].get("filename")
        return None

    async def save_image(self, worker_url: str, filename: str, local_name: str) -> str | None:
        """Fetch and save image locally from the worker, optionally upload to S3"""
        os.makedirs("generated_images", exist_ok=True)
        return await self._fetch_image(worker_url, filename, local_name)

    def _upload_to_s3(self, local_path: str, s3_key: str) -> str | None:
        """Upload a local file to S3 and return the S3 URL"""
        if not self.s3_client:
            return None
        
        try:
            self.s3_client.upload_file(
                local_path,
                S3_BUCKET_NAME,
                s3_key,
                ExtraArgs={"ContentType": "image/png"}
            )
            s3_url = f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{s3_key}"
            print(f"  ☁️  Uploaded to S3: {s3_key}")
            return s3_url
        except Exception as e:
            log.error(f"Failed to upload to S3: {e}")
            return None

    async def _fetch_image(self, worker_url: str, filename: str, local_name: str) -> str | None:
        """Fetch image from worker's /view endpoint and save locally"""
        if not worker_url:
            return None
            
        try:
            url = f"{worker_url}/view"
            params = {"filename": filename, "type": "output"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, ssl=False) as resp:
                    if resp.status == 200:
                        path = f"generated_images/{local_name}"
                        image_data = await resp.read()
                        with open(path, "wb") as f:
                            f.write(image_data)
                        print(f"  💾 Saved: {path}")
                        
                        # Upload to S3 if enabled
                        if self.upload_s3 and self.s3_client:
                            s3_key = f"comfyui/{local_name}"
                            self._upload_to_s3(path, s3_key)
                        
                        return path
            return None
        except Exception:
            return None

    async def demo_prompt(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int | None,
    ):
        """Demo: Generate image from text prompt"""
        print("=" * 60)
        print("COMFYUI TEXT-TO-IMAGE DEMO")
        print("=" * 60)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Size: {width}x{height}, Steps: {steps}, Seed: {seed}")
        print("\n🎨 Generating image...")

        response = await call_generate(
            self.client,
            endpoint_name=self.endpoint_name,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )

        print("\n✅ Generation complete!")

        # Get worker URL for fetching images
        worker_url = response.get("url", "")
        print(f"Worker URL: {worker_url}")

        # Fetch and save image
        if "response" in response:
            filename = self.extract_filename(response["response"])
            if filename:
                path = await self.save_image(worker_url, filename, f"comfy_{seed}.png")
                if not path:
                    print("❌ Failed to fetch image")
            else:
                print("❌ No image in response")
        else:
            print("❌ Unexpected response format")

    async def demo_workflow(self, workflow_file: str):
        """Demo: Generate using custom workflow file"""
        print("=" * 60)
        print("COMFYUI CUSTOM WORKFLOW DEMO")
        print("=" * 60)

        if not os.path.exists(workflow_file):
            log.error(f"Workflow file not found: {workflow_file}")
            return

        with open(workflow_file, "r") as f:
            workflow_json = json.load(f)

        print(f"Workflow: {workflow_file}")
        print("\n🎨 Generating...")

        response = await call_generate_workflow(
            self.client,
            endpoint_name=self.endpoint_name,
            workflow_json=workflow_json,
        )

        print("\n✅ Generation complete!")

        worker_url = response.get("url", "")

        if "response" in response:
            filename = self.extract_filename(response["response"])
            if filename:
                path = await self.save_image(worker_url, filename, "workflow.png")
                if not path:
                    print("❌ Failed to fetch image")
            else:
                print("❌ No image in response")
        else:
            print("❌ Unexpected response format")


# ---------------------- CLI ----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast ComfyUI-JSON Demo (Serverless SDK)")
    p.add_argument("--endpoint", default=ENDPOINT_NAME, help=f"Vast endpoint name (default: {ENDPOINT_NAME})")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, metavar="TEXT",
                   help=f"Prompt text (default: '{DEFAULT_PROMPT[:30]}...')")
    p.add_argument("--workflow", type=str, metavar="FILE", help="Use custom workflow JSON file instead")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Image width (default: {DEFAULT_WIDTH})")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Image height (default: {DEFAULT_HEIGHT})")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Steps (default: {DEFAULT_STEPS})")
    p.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    p.add_argument("--s3", action="store_true", 
                   help="Upload generated images to S3 (requires S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY env vars)")
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    print("=" * 60)
    print(f"Using endpoint: {args.endpoint}")
    if args.s3:
        print(f"S3 upload: enabled (bucket: {S3_BUCKET_NAME})")

    try:
        async with Serverless() as client:
            demo = APIDemo(client, args.endpoint, upload_s3=args.s3)

            if args.workflow:
                await demo.demo_workflow(workflow_file=args.workflow)
            else:
                await demo.demo_prompt(
                    prompt=args.prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    seed=args.seed,
                )

    except AttributeError as e:
        if "API key" in str(e):
            log.error("API key missing. Set VAST_API_KEY environment variable.")
        else:
            log.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
