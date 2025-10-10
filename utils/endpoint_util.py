import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class Endpoint:
    """
    Utility class for handling endpoint operations.
    """

    @staticmethod
    def get_endpoint_info(
        endpoint_name: str, account_api_key: str, instance: str
    ) -> Optional[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {account_api_key}"}
        url = f"{Endpoint.get_server_url(instance)}?autoscaler_instance={instance}"
        # Retry a few times to smooth over transient propagation/network delays
        for attempt in range(4):
            try:
                response = requests.get(url, headers=headers, timeout=8)
                if response.status_code != 200:
                    # brief backoff and retry
                    time.sleep(0.3 * (attempt + 1))
                    continue
                try:
                    data = response.json()
                except Exception:
                    # JSON parse failed; backoff and retry
                    time.sleep(0.3 * (attempt + 1))
                    continue
                result = data.get("results", []) if isinstance(data, dict) else []
                endpoint = next(
                    (item for item in result if item.get("endpoint_name") == endpoint_name),
                    None,
                )
                if endpoint and endpoint.get("id") and endpoint.get("api_key"):
                    return {"id": endpoint.get("id"), "api_key": endpoint.get("api_key")}
            except Exception:
                # network or other transient error; retry
                time.sleep(0.3 * (attempt + 1))
        return None

    @staticmethod
    def get_autoscaler_server_url(instance: str) -> str:
        endpoints = {
            "alpha": "run-alpha",
            "candidate": "run-candidate",
            "prod": "run",
        }
        host = endpoints.get(instance)
        if host:
            return f"https://{host}.vast.ai/"
        return "http://localhost:8080"

    @staticmethod
    def get_server_url(instance: str) -> str:
        endpoints = {
            "alpha": "alpha",
            "candidate": "candidate",
            "prod": "console",
        }
        host = endpoints.get(instance, "alpha")
        return f"https://{host}.vast.ai/api/v0/endptjobs/"

    @staticmethod
    def get_endpoint_api_key(
        endpoint_name: str, account_api_key: str, instance: str
    ) -> Optional[str]:
        """
        Fetch endpoint API key from VastAI console following the healthcheck pattern.

        Args:
            endpoint_name: Name of the endpoint
            account_api_key: Account API key for authentication

        Returns:
            Endpoint API key if successful, None otherwise
        """
        headers = {"Authorization": f"Bearer {account_api_key}"}

        try:
            log.debug(f"Fetching endpoint API key for endpoint: {endpoint_name}")
            response = requests.get(
                f"{Endpoint.get_server_url(instance)}?autoscaler_instance={instance}",
                headers=headers,
                timeout=8,
            )

            if response.status_code != 200:
                error_msg = f"Failed to fetch endpoint API key: {response.status_code} - {response.text}"
                log.debug(error_msg)
                return None

            try:
                data = response.json()
            except Exception as e:
                log.debug(f"Failed to parse JSON response: {e}")
                return None

            result = data.get("results", [])

            endpoint: Optional[Dict[str, Any]] = next(
                (item for item in result if item.get("endpoint_name") == endpoint_name),
                None,
            )
            if not endpoint:
                error_msg = f"Endpoint '{endpoint_name}' not found."
                log.debug(error_msg)
                return None

            endpoint_api_key = endpoint.get("api_key")
            if not endpoint_api_key:
                error_msg = f"API key for endpoint '{endpoint_name}' not found."
                log.debug(error_msg)
                return None

            log.debug(f"Successfully retrieved API key for endpoint: {endpoint_name}")
            return endpoint_api_key

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error while fetching endpoint API key: {e}"
            log.debug(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error while fetching endpoint API key: {e}"
            log.debug(error_msg)
            return None
