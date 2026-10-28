
import { Link } from "react-router-dom";
import {
  ArrowRight,
  ShieldCheck,
  ScanSearch,
  BrainCircuit,
  Image as ImageIcon,
  CheckCircle2,
} from "lucide-react";

const features = [
  {
    icon: ScanSearch,
    title: "Image Analysis",
    description:
      "Submit an image for analysis through the SecureSight detection pipeline.",
  },
  {
    icon: BrainCircuit,
    title: "Hybrid Detection",
    description:
      "Combine CNN classification with diffusion-based features.",
  },
  {
    icon: ShieldCheck,
    title: "Transparent Results",
    description:
      "View the prediction and available confidence scores.",
  },
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="status-dot" />
            AI IMAGE AUTHENTICITY PLATFORM
          </div>

          <h1>
            See beyond the image.
            <br />
            <span className="gradient-text">
              Discover what's real.
            </span>
          </h1>

          <p className="hero-description">
            SecureSight analyzes digital images using a hybrid
            CNN and diffusion-based approach to help identify
            AI-generated content.
          </p>

          <div className="hero-actions">
            <Link to="/detect" className="button-primary">
              Analyze an Image <ArrowRight size={18} />
            </Link>
            <a href="#features" className="button-secondary">
              Explore Features
            </a>
          </div>

          <div className="trust-note">
            <CheckCircle2 size={16} />
            Built with deep learning and image analysis
          </div>
        </div>

        <div className="hero-visual">
          <div className="visual-glow" />
          <div className="analysis-card">
            <div className="analysis-top">
              <div>
                <span className="small-label">SECURESIGHT</span>
                <h3>Detection Pipeline</h3>
              </div>
              <span className="live-indicator">READY</span>
            </div>

            <div className="scan-illustration">
              <div className="scan-circle">
                <div className="scan-inner">
                  <ShieldCheck size={66} strokeWidth={1.2} />
                </div>
                <span className="orbit orbit-one" />
                <span className="orbit orbit-two" />
              </div>
              <div className="scan-line" />
            </div>

            <div className="pipeline">
              <div className="pipeline-step">
                <span className="pipeline-icon">
                  <ImageIcon size={18} />
                </span>
                <div>
                  <strong>Image Input</strong>
                  <small>Upload & preprocessing</small>
                </div>
                <CheckCircle2 size={17} className="success-icon" />
              </div>

              <div className="pipeline-step">
                <span className="pipeline-icon">
                  <BrainCircuit size={18} />
                </span>
                <div>
                  <strong>Hybrid Analysis</strong>
                  <small>CNN + diffusion features</small>
                </div>
                <span className="pipeline-arrow">→</span>
              </div>

              <div className="pipeline-step">
                <span className="pipeline-icon">
                  <ScanSearch size={18} />
                </span>
                <div>
                  <strong>Prediction</strong>
                  <small>Classification result</small>
                </div>
                <span className="pipeline-arrow">→</span>
              </div>
            </div>

            <p className="disclaimer">
              Results depend on model performance and image quality.
            </p>
          </div>
        </div>
      </section>

      <section className="features-section" id="features">
        <div className="section-heading">
          <span className="eyebrow">WHAT SECURESIGHT DOES</span>
          <h2>Built for image authenticity analysis.</h2>
          <p>
            A unified interface for submitting images and
            reviewing model-generated predictions.
          </p>
        </div>

        <div className="feature-grid">
          {features.map(({ icon: Icon, title, description }) => (
            <article className="feature-card" key={title}>
              <div className="feature-icon">
                <Icon size={23} />
              </div>
              <h3>{title}</h3>
              <p>{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="bottom-cta">
        <div>
          <span className="eyebrow">READY TO GET STARTED?</span>
          <h2>Put an image to the test.</h2>
          <p>Upload an image to begin the analysis workflow.</p>
        </div>
        <Link to="/detect" className="button-primary">
          Start Detection <ArrowRight size={18} />
        </Link>
      </section>

      <footer className="footer">
        <span>© {new Date().getFullYear()} SecureSight</span>
        <span>AI-Generated Image Detection</span>
      </footer>
    </main>
  );
}
