export default function Hero() {
  return (
    <section id="home" className="text-center py-12 px-6">
      <div className="inline-block px-4 py-1.5 mb-6 text-sm font-semibold tracking-wide text-blue-600 uppercase bg-blue-50 rounded-full">
        Next-Gen Authenticity
      </div>
      <h1 className="text-5xl md:text-9xl font-extrabold mb-6 tracking-tight">
        Aletheia
      </h1>
      <p className="text-xl md:text-2xl font-medium mb-4 text-blue-800">
        Truth Revealed Through Innovation
      </p>
      <p className="max-w-2xl mx-auto text-blue-600/80 leading-relaxed text-lg">
        Verify image authenticity using cutting-edge AI before you trust or share content online. 
        Identify deepfakes and manipulations in seconds.
      </p>
      <div className="mt-10 flex flex-wrap justify-center gap-4">
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-2xl font-bold transition-all transform hover:scale-105 shadow-lg">
          Start Verification
        </button>
        <button className="bg-white hover:bg-blue-50 border border-blue-200 px-8 py-4 rounded-2xl font-bold transition-all">
          Learn More
        </button>
      </div>
    </section>
  );
}