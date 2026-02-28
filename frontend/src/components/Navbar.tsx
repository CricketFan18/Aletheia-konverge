export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-white/70 backdrop-blur-md border-b border-blue-100">
      <div className="max-w-7xl mx-auto px-6 py-8 flex justify-between items-center">
        <div className="font-bold text-3xl tracking-tight flex items-center gap-2">
          <span className="text-blue-600">🛡</span> Aletheia
        </div>
        <div className="hidden md:flex items-center space-x-8 text-sm font-medium">
          <a href="#home" className="hover:text-blue-600 text-2xl transition-colors">Home</a>
          <a href="#verify" className="hover:text-blue-600 text-2xl transition-colors">Verify</a>
          <a href="#about" className="hover:text-blue-600 text-2xl transition-colors">About</a>
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl transition-all shadow-md shadow-blue-200">
            Get Started
          </button>
        </div>
      </div>
    </nav>
  );
}