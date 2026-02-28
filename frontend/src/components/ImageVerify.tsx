import { useState, useRef, type ChangeEvent } from "react";

interface VerifyResult {
  status: string;
  verdict: {
    label: string;
    confidence: number;
  };
  evidence: {
    heatmap_image: string;
    metadata: string;
  };
}

export default function ImageVerify() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const maxFileSize = 10 * 1024 * 1024;
    if (selectedFile.size > maxFileSize) {
      alert("File size exceeds 10MB. Please choose a smaller file.");
      e.target.value = "";
      return;
    }

    setFile(selectedFile);
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    // MUST match FastAPI parameter name: file: UploadFile = File(...)
    formData.append("file", selectedFile);

    try {
      const response = await fetch("https://interstrial-epithelial-anaya.ngrok-free.dev/api/analyze", {
        method: "POST",
        body: formData,
      });
      const contentType = response.headers.get("content-type");

      
      if (!response.ok) {
        const text = await response.text();
        console.error("Server error response:", text);
        throw new Error(`Server Error: ${response.status}`);
      }

      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        console.error("Non-JSON response:", text);
        throw new Error("Invalid response format (not JSON)");
      }

      const data: VerifyResult = await response.json();
      console.log("API Response:", data);

      setResult(data);
    } catch (err) {
      console.error("Verification failed:", err);
      setError("Failed to analyze image. Check backend or API URL.");
    } finally {
      setLoading(false);
    }
  };

  const confidencePercent = result?.verdict?.confidence ?? 0;
  const label = result?.verdict?.label ?? "";

  return (
    <section id="verify" className="max-w-7xl mx-auto px-6 py-20">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">Image Authenticity Verifier</h2>
        <p className="text-blue-600">
          Secure, private, and powered by neural networks.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 bg-white/40 backdrop-blur-sm rounded-[2.5rem] p-8 md:p-12 border border-white shadow-2xl">
        {/* Upload Box */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className="cursor-pointer border-2 border-dashed border-blue-200 bg-blue-50/50 rounded-3xl p-12 text-center flex flex-col items-center justify-center group hover:border-blue-400 transition-colors"
        >
          <input
            type="file"
            className="hidden"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp"
          />

          <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            📁
          </div>

          <p className="text-xl font-bold mb-2">
            {file ? file.name : "Upload Image"}
          </p>

          <p className="text-sm text-blue-500 mb-8 font-medium">
            JPG, PNG or WEBP • Up to 10MB
          </p>

          <button
            type="button"
            className="bg-blue-600 hover:bg-blue-700 text-white px-10 py-3.5 rounded-xl font-bold shadow-lg shadow-blue-200 transition-all"
          >
            {loading ? "Processing..." : "Select File"}
          </button>
        </div>

        {/* Result Panel */}
        <div className="bg-blue-900/5 rounded-3xl flex flex-col items-center justify-center p-8 border border-blue-100">
          {loading ? (
            <div className="text-center text-blue-400">
              <p className="text-5xl mb-4">⚙️</p>
              <p className="font-medium">Analyzing pixels...</p>
            </div>
          ) : error ? (
            <div className="text-center text-red-500">
              <p className="text-xl font-bold mb-2">Error</p>
              <p>{error}</p>
            </div>
          ) : result ? (
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-900 mb-2">
                Analysis Complete
              </p>

              <div className="p-4 bg-white rounded-xl shadow-sm border border-blue-100 mb-4">
                <p className="text-sm text-gray-500">Verdict</p>
                <p
                  className={`text-2xl font-black ${
                    label === "AUTHENTIC" ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {label}
                </p>
              </div>

              <div className="p-4 bg-white rounded-xl shadow-sm border border-blue-100">
                <p className="text-sm text-gray-500">Confidence</p>
                <p className="text-3xl font-black text-blue-600">
                  {confidencePercent.toFixed(2)}%
                </p>
              </div>

              {result.evidence?.heatmap_image && (
                <img
                  src={result.evidence.heatmap_image}
                  alt="ELA Heatmap"
                  className="mt-6 rounded-xl border shadow-md max-h-64 object-contain"
                />
              )}
            </div>
          ) : (
            <div className="text-center text-blue-400">
              <p className="text-5xl mb-4">🔍</p>
              <p className="font-medium">Waiting for upload...</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}