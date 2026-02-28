export default function Features() {
const features = [
    { title: "GAN-Focused Detection", desc: "Our neural networks identify microscopic inconsistencies in pixel patterns and facial warping.", icon: "🤖" },
    { title: "Lightning Fast API", desc: "Local EfficientNet processing delivers comprehensive visual and metadata results instantly.", icon: "⚡" },
    { title: "Metadata Fallback", desc: "Deep dive into EXIF data structure. Automatically flags images stripped of camera data by modern AI generators.", icon: "📊" },
  ];

  return (
    <section className="py-24 px-6 max-w-7xl mx-auto">
      <h2 className="text-4xl font-bold text-center mb-16 tracking-tight">Powerful Features</h2>
      <div className="grid md:grid-cols-3 gap-8">
        {features.map((f) => (
          <div key={f.title} className="bg-white p-10 rounded-4xl shadow-sm border border-blue-50 hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-6">{f.icon}</div>
            <h3 className="font-bold text-xl mb-3">{f.title}</h3>
            <p className="text-blue-600/80 leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}