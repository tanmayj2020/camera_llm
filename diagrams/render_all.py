#!/usr/bin/env python3
"""Render all Mermaid .mmd files to high-res PNG using Kroki API (no Node required)."""

import base64, json, os, sys, ssl, zlib, urllib.request, urllib.error, glob

DIAGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
KROKI_URL = "https://kroki.io/mermaid/png"

# Create SSL context that skips cert verification (corporate proxy issue)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

def render_mmd_to_png(mmd_path: str, png_path: str) -> bool:
    """Send Mermaid source to Kroki API and save the returned PNG."""
    with open(mmd_path, "r") as f:
        mmd_source = f.read()

    # Kroki accepts POST with JSON body
    payload = json.dumps({"diagram_source": mmd_source}).encode("utf-8")
    req = urllib.request.Request(
        KROKI_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "image/png"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx) as resp:
            png_data = resp.read()
        with open(png_path, "wb") as f:
            f.write(png_data)
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    mmd_files = sorted(glob.glob(os.path.join(DIAGRAMS_DIR, "*.mmd")))
    if not mmd_files:
        print("No .mmd files found!")
        sys.exit(1)

    print(f"Found {len(mmd_files)} Mermaid diagrams. Rendering via Kroki API...\n")

    success = 0
    failed = 0
    for mmd_path in mmd_files:
        name = os.path.basename(mmd_path)
        png_name = name.replace(".mmd", ".png")
        png_path = os.path.join(DIAGRAMS_DIR, png_name)
        print(f"  Rendering {name} → {png_name} ...", end=" ", flush=True)
        if render_mmd_to_png(mmd_path, png_path):
            size_kb = os.path.getsize(png_path) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success += 1
        else:
            print("FAILED")
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed out of {len(mmd_files)}")


if __name__ == "__main__":
    main()
