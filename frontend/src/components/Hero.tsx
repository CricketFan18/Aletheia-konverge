export default function Hero() {
  return (
    <section id="home" className="text-center py-12 px-6">
      <div className="inline-block px-4 py-1.5 mb-6 text-sm font-semibold tracking-wide text-blue-600 uppercase bg-blue-50 rounded-full">
        GAN & Deepfake Authenticity
      </div>
      <h1 className="text-5xl md:text-9xl font-extrabold mb-6 tracking-tight">
        Aletheia
      </h1>
      <p className="text-xl md:text-2xl font-medium mb-4 text-blue-800">
        Truth Revealed Through Innovation
      </p>
      <p className="max-w-2xl mx-auto text-blue-600/80 leading-relaxed text-lg">
        Verify image authenticity against facial manipulation and GAN-based deepfakes using cutting-edge AI. 
        Identify hidden tampering in seconds before you trust or share.
      </p>
      <div className="mt-10 flex flex-wrap justify-center gap-4">
        <a href="#verify" className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-2xl font-bold transition-all transform hover:scale-105 shadow-lg">
          Start Verification
        </a>
      </div>
    </section>
  );
}