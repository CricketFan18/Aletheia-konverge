import { useState, useRef, type ChangeEvent, useEffect } from "react";

interface VerifyResult {
  status: string;
  verdict: {
    label: string;
    confidence: number;
  };
  evidence: {
    heatmap_image: string;
    metadata: {
      status: string;
      data: any;
    };
  };
}

// 1. Define the demo samples mapping to your public/samples folder
const DEMO_SAMPLES = [
  { id: 1, label: "REAL", src: "/samples/real1.jpg", name: "real_demo_1.jpg" },
  { id: 2, label: "REAL", src: "/samples/real2.jpg", name: "real_demo_2.jpg" },
  { id: 3, label: "REAL", src: "/samples/real3.jpg", name: "real_demo_3.jpg" },
  { id: 4, label: "FAKE", src: "/samples/fake1.jpg", name: "fake_gan_1.jpg" },
  { id: 5, label: "FAKE", src: "/samples/fake2.jpg", name: "fake_gan_2.jpg" },
  { id: 6, label: "FAKE", src: "/samples/fake3.jpg", name: "fake_gan_3.jpg" },
];

export default function ImageVerify() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
    };
  }, [imagePreview]);

  // 2. Centralized analysis function for both uploads and sample clicks
  const analyzeFile = async (selectedFile: File) => {
    setFile(selectedFile);
    
    const objectUrl = URL.createObjectURL(selectedFile);
    setImagePreview(objectUrl);
    
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      const contentType = response.headers.get("content-type");

      if (!response.ok) {
        throw new Error(`Server Error: ${response.status}`);
      }

      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Invalid response format (not JSON)");
      }

      const data: VerifyResult = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Verification failed:", err);
      setError("Failed to analyze image. Check backend or API URL.");
    } finally {
      setLoading(false);
    }
  };

  // Handler for manual uploads
  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const maxFileSize = 10 * 1024 * 1024;
    if (selectedFile.size > maxFileSize) {
      alert("File size exceeds 10MB. Please choose a smaller file.");
      e.target.value = "";
      return;
    }

    await analyzeFile(selectedFile);
  };

  // 3. Handler for demo gallery clicks
  const handleSampleClick = async (sample: typeof DEMO_SAMPLES[0]) => {
    try {
      setLoading(true);
      // Fetch the image from the public folder and convert it to a File object
      const response = await fetch(sample.src);
      const blob = await response.blob();
      const demoFile = new File([blob], sample.name, { type: blob.type });
      
      await analyzeFile(demoFile);
    } catch (err) {
      console.error("Failed to load demo sample", err);
      setError("Could not load demo image. Make sure it exists in the public/samples folder.");
      setLoading(false);
    }
  };

  const confidencePercent = result?.verdict?.confidence ?? 0;
  const label = result?.verdict?.label ?? "";
  const metadataStatus = result?.evidence?.metadata?.status;
  const showMetadataWarning = label === "AUTHENTIC" && metadataStatus === "Missing";

  return (
    <section id="verify" className="max-w-7xl mx-auto px-6 py-20">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">GAN Authenticity Verifier</h2>
        <p className="text-blue-600 mb-4">
          Secure, private, and powered by neural networks.
        </p>
        
        <div className="inline-block bg-blue-50 border border-blue-100 rounded-lg px-4 py-2 text-sm text-blue-800">
          <strong>MVP Note:</strong> Current model weights (v1.0) are optimized for Generative Adversarial Networks (GANs) and facial warping. <br/>
          <em>Diffusion model support (Midjourney/DALL-E) slated for v2.0 roadmap.</em>
        </div>
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
              <p className="text-5xl mb-4 animate-spin w-12 h-12 mx-auto border-4 border-blue-400 border-t-transparent rounded-full"></p>
              <p className="font-medium">Analyzing pixels & metadata...</p>
            </div>
          ) : error ? (
            <div className="text-center text-red-500">
              <p className="text-xl font-bold mb-2">Error</p>
              <p>{error}</p>
            </div>
          ) : result ? (
            <div className="text-center w-full">
              <p className="text-2xl font-bold text-blue-900 mb-4">
                Analysis Complete
              </p>

              <div className="flex justify-center gap-4 mb-6">
                <div className="p-4 bg-white rounded-xl shadow-sm border border-blue-100 flex-1">
                  <p className="text-sm text-gray-500">Verdict</p>
                  <p
                    className={`text-2xl font-black ${
                      label === "AUTHENTIC" ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {label}
                  </p>
                </div>

                <div className="p-4 bg-white rounded-xl shadow-sm border border-blue-100 flex-1">
                  <p className="text-sm text-gray-500">Confidence</p>
                  <p className="text-2xl font-black text-blue-600">
                    {confidencePercent.toFixed(2)}%
                  </p>
                </div>
              </div>

              {showMetadataWarning && (
                <div className="mb-6 p-4 bg-yellow-50 border border-yellow-300 text-yellow-800 rounded-xl text-sm font-semibold text-left shadow-sm">
                  ⚠️ META-DATA ALERT: Visuals appear authentic, but zero EXIF camera data was found. High probability of modern synthetic generation (Diffusion) or heavy digital scrubbing.
                </div>
              )}

              <div className="bg-white p-4 rounded-xl shadow-sm border border-blue-100 w-full">
                <p className="text-xs text-gray-500 mb-3 uppercase font-bold tracking-wider text-center">Visual Analysis</p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  
                  {imagePreview && (
                    <div className="flex-1 flex flex-col items-center w-full">
                      <span className="text-[10px] text-blue-400 font-bold mb-1.5 uppercase">Original Upload</span>
                      <img
                        src={imagePreview}
                        alt="Original Upload"
                        className="rounded-lg border border-gray-200 shadow-sm max-h-40 w-full object-cover"
                      />
                    </div>
                  )}

                  {result.evidence?.heatmap_image && (
                    <div className="flex-1 flex flex-col items-center w-full">
                      <span className="text-[10px] text-blue-400 font-bold mb-1.5 uppercase">ELA Heatmap</span>
                      <img
                        src={result.evidence.heatmap_image}
                        alt="ELA Heatmap"
                        className="rounded-lg border border-gray-200 shadow-sm max-h-40 w-full object-cover"
                      />
                    </div>
                  )}
                  
                </div>
              </div>

            </div>
          ) : (
            <div className="text-center text-blue-400">
              <p className="text-5xl mb-4">🔍</p>
              <p className="font-medium">Waiting for upload...</p>
            </div>
          )}
        </div>
      </div>

      {/* 4. Live Demo Quick-Test Gallery */}
      <div className="mt-8 bg-white/60 backdrop-blur-sm rounded-3xl p-8 border border-white shadow-lg w-full">
        <div className="text-center mb-6">
          <h3 className="text-2xl font-bold text-blue-900">Live Demo Gallery</h3>
          <p className="text-blue-600">Click a sample to instantly run it through the PyTorch engine.</p>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSampleClick(sample)}
              disabled={loading}
              className="group relative rounded-xl overflow-hidden border-2 border-transparent hover:border-blue-500 focus:outline-none focus:ring-4 focus:ring-blue-200 transition-all disabled:opacity-50"
            >
              <img 
                src={sample.src} 
                alt={sample.name} 
                className="w-full h-24 object-cover transform group-hover:scale-110 transition-transform duration-300"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-blue-900/80 to-transparent flex flex-col justify-end p-2">
                <span className={`text-[10px] font-black tracking-widest ${sample.label === 'REAL' ? 'text-green-400' : 'text-red-400'}`}>
                  {sample.label}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

    </section>
  );
}