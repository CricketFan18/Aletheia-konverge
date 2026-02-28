export default function Footer() {
  return (
    <footer className="bg-white border-t border-blue-50 py-16 text-center">
      <div className="max-w-7xl mx-auto px-6">
        <p className="font-bold text-2xl mb-4 tracking-tight">Aletheia</p>
        <p className="text-blue-600 max-w-md mx-auto mb-8">
          The global standard for AI-powered image verification and digital trust.
        </p>
        {/* <div className="text-xs text-blue-400 font-medium">
          © {new Date().getFullYear()} Aletheia Systems Inc. All rights reserved.
        </div> */}
      </div>
    </footer>
  );
}