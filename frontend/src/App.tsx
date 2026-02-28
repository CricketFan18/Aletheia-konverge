import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import ImageVerify from "./components/ImageVerify";
import Features from "./components/Features";
// import HowItWorks from "./components/HowItWorks";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="bg-linear-to-br from-blue-50 via-white to-blue-100 text-blue-900 selection:bg-blue-200">
      <Navbar />
      <main>
        <Hero />
        <ImageVerify />
        <Features />
        {/* <HowItWorks /> */}
      </main>
      <Footer />
    </div>
  );
}