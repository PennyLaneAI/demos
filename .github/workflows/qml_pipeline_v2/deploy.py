import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from requests_auth_aws_sigv4 import AWSSigV4

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MAX_RESPONSE_BODY_LOG_CHARS = 1000
DEPLOYMENT_URL_TEMPLATE = "{endpoint_url}/demo/{slug}?pr_number={pr_number}"

parser = argparse.ArgumentParser(
    prog="deploy",
    description="Deploy demo zip files to the QML website.",
)

parser.add_argument(
    "paths", help="The pattern to search for demo zip files.", type=Path, nargs="+"
)
parser.add_argument(
    "-pr",
    "--pr-number",
    type=str,
    default=None,
    help="Whether to deploy the demo as a pull request preview or as a main branch deployment. Use '0' for the main version (prod), and a PR number greater than '0' for a preview version.",
)


def _response_body_snippet(response: requests.Response | None) -> str:
    """Return a best-effort response body snippet for debugging."""
    if response is None:
        return ""

    body = response.text
    if len(body) <= MAX_RESPONSE_BODY_LOG_CHARS:
        return body

    return (
        body[:MAX_RESPONSE_BODY_LOG_CHARS]
        + "\n... [truncated] ..."
        + f"\n(total chars: {len(body)})"
    )


def main():
    args = parser.parse_args()
    pr_number: str = args.pr_number
    paths: list[Path] = args.paths

    session = requests.Session()
    session.auth = AWSSigV4("execute-api", region="us-east-1")

    endpoint_url = os.environ["DEPLOYMENT_ENDPOINT_URL"]

    for path in paths:
        if not path.exists():
            logger.error("Path '%s' does not exist.", path)
            sys.exit(1)
        elif not (path.is_file() and path.suffix == ".zip"):
            logger.error("Path '%s' is not a zip file.", path)
            sys.exit(1)

        slug = path.stem
        url = DEPLOYMENT_URL_TEMPLATE.format(
            endpoint_url=endpoint_url, slug=slug, pr_number=pr_number
        )
        with open(path, "rb") as f:
            logger.info("Deploying '%s' to '%s'", path, url)
            try:
                session.put(url, files={"file": f}).raise_for_status()
            except requests.HTTPError as exc:
                response = exc.response
                if response is not None:
                    logger.error(
                        "Failed to deploy '%s' to '%s' (status_code=%s, reason=%s)",
                        path,
                        url,
                        response.status_code,
                        response.reason,
                        exc_info=True,
                    )
                    logger.error(
                        "Response body:\n%s",
                        _response_body_snippet(response),
                        exc_info=True,
                    )
                else:
                    logger.error(
                        "Failed to deploy '%s' to '%s'", path, url, exc_info=True
                    )
                sys.exit(1)
            except requests.RequestException:
                logger.error(
                    "Request error while deploying '%s' to '%s'",
                    path,
                    url,
                    exc_info=True,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
