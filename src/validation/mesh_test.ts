/**
 * End-to-end mesh pipeline test.
 * Reads Python pickle cache from pipeline run, produces .glb via TS mesh pipeline.
 *
 * Run: deno run --allow-read --allow-write src/validation/mesh_test.ts
 */
import { meshFromVoxels } from "../mesh/pipeline.ts";

async function main() {
  console.log("Mesh Pipeline Test: Python cache → TS mesh → GLB\n");

  // Read Python pickle cache (binary format — we need to parse it)
  // The cache is a pickle file. We'll use the Python subprocess to extract data.
  // Alternatively, run a quick Python script to dump coords+attrs as .npy

  const cachePath = "reference-images/T-80BVM.glb.cache.pkl";
  const outPath = "reference-images/T-80BVM-ts.glb";

  try {
    // Use Python to extract cache data to raw binary
    const pythonPath = "TRELLIS.2-stableprojectorz/venv/Scripts/python.exe";
    const extractCmd = new Deno.Command(pythonPath, {
      args: ["-c", `
import pickle, sys, os
cache_path = r"${Deno.cwd().replace(/\\/g, "/")}/${cachePath}"
try:
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    import numpy as np
    coords = data["coords"].numpy().astype("int32")
    attrs = data["attrs"].numpy().astype("float32")

    # Save as raw binary
    coords_path = r"${Deno.cwd().replace(/\\/g, "/")}/reference-images/_coords.bin"
    attrs_path = r"${Deno.cwd().replace(/\\/g, "/")}/reference-images/_attrs.bin"

    with open(coords_path, "wb") as f:
        f.write(coords.tobytes())
    with open(attrs_path, "wb") as f:
        f.write(attrs.tobytes())

    print(f"coords: {coords.shape}, attrs: {attrs.shape}")
    print(f"Written {coords.nbytes} + {attrs.nbytes} bytes")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
`],
      cwd: Deno.cwd(),
    });

    const output = await extractCmd.output();
    const outStr = new TextDecoder().decode(output.stdout);
    console.log(outStr);

    if (output.code !== 0) {
      console.error("Failed to extract cache:", new TextDecoder().decode(output.stderr));
      Deno.exit(1);
    }

    // Read the raw binary data
    const coordsRaw = Deno.readFileSync("reference-images/_coords.bin");
    const attrsRaw = Deno.readFileSync("reference-images/_attrs.bin");

    const coords = new Int32Array(coordsRaw.buffer);
    const attrs = new Float32Array(attrsRaw.buffer);

    console.log(`coords: ${coords.length} elements (${coords.length / 3} voxels)`);
    console.log(`attrs: ${attrs.length} elements (${attrs.length / 6} channels)`);

    // Run mesh pipeline
    console.log("\nRunning TS mesh pipeline...");
    const glb = meshFromVoxels(
      { coords, attrs, attrLayout: { base_color: [0, 3], metallic: 3, roughness: 4, alpha: 5 } },
      { verbose: true },
    );

    // Write GLB
    Deno.writeFileSync(outPath, new Uint8Array(glb));
    console.log(`\nSaved ${outPath} (${glb.byteLength} bytes)`);

    // Verify file exists and has content
    const stat = Deno.statSync(outPath);
    console.log(`File size: ${stat.size} bytes`);
    console.log("VALID GLB: " + (stat.size > 1000 ? "YES" : "NO (too small)"));

    // Cleanup
    try { Deno.removeSync("reference-images/_coords.bin"); } catch { /* ok */ }
    try { Deno.removeSync("reference-images/_attrs.bin"); } catch { /* ok */ }
  } catch (e) {
    console.error("Error:", e);
    Deno.exit(1);
  }
}

main();
