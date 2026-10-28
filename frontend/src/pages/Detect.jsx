
import { useRef, useState } from "react";
import {
    Upload,
    Image as ImageIcon,
    X,
    LoaderCircle,
    ShieldCheck,
} from "lucide-react";
import { analyzeImage } from "../services/api";

export default function Detect() {
    const inputRef = useRef(null);
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState("");
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    function selectFile(selected) {
        if (!selected) return;

        if (!selected.type.startsWith("image/")) {
            setError("Please select a valid image file.");
            return;
        }

        if (selected.size > 15 * 1024 * 1024) {
            setError("Please select an image smaller than 15 MB.");
            return;
        }

        setFile(selected);
        setPreview(URL.createObjectURL(selected));
        setResult(null);
        setError("");
    }

    function clearFile() {
        if (preview) URL.revokeObjectURL(preview);
        setFile(null);
        setPreview("");
        setResult(null);
        setError("");
        if (inputRef.current) inputRef.current.value = "";
    }

    async function handleAnalyze() {
        if (!file) return;

        setLoading(true);
        setError("");
        setResult(null);

        try {
            const data = await analyzeImage(file);
            setResult(data);
        } catch (err) {
            setError(
                err.message ||
                "Analysis failed. Check that the FastAPI backend is running."
            );
        } finally {
            setLoading(false);
        }
    }

    const label =
        result?.prediction ??
        result?.label ??
        result?.class ??
        result?.result;

    const isFake =
        typeof label === "string" &&
        ["fake", "ai", "ai-generated", "artificial"].includes(
            label.toLowerCase()
        );

    return (
        <main className="detect-page">
            <div className="detect-heading">
                <span className="eyebrow">SECURESIGHT ANALYZER</span>
                <h1>Analyze an image</h1>
                <p>
                    Upload an image to submit it to the SecureSight
                    detection backend.
                </p>
            </div>

            <div className="detect-layout">
                <section className="upload-panel">
                    <div className="panel-heading">
                        <div>
                            <h2>Image Input</h2>
                            <p>Select an image for analysis.</p>
                        </div>
                        <ImageIcon size={23} />
                    </div>

                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        hidden
                        onChange={(e) => selectFile(e.target.files?.[0])}
                    />

                    {!preview ? (
                        <button
                            type="button"
                            className="drop-zone"
                            onClick={() => inputRef.current?.click()}
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                                e.preventDefault();
                                selectFile(e.dataTransfer.files?.[0]);
                            }}
                        >
                            <span className="upload-icon">
                                <Upload size={27} />
                            </span>
                            <strong>Drop your image here</strong>
                            <span>or click to browse your files</span>
                            <small>Image files · Maximum 15 MB</small>
                        </button>
                    ) : (
                        <div className="preview-container">
                            <img src={preview} alt="Selected image preview" />
                            <button
                                type="button"
                                className="remove-image"
                                onClick={clearFile}
                                aria-label="Remove selected image"
                            >
                                <X size={18} />
                            </button>
                        </div>
                    )}

                    {file && (
                        <div className="selected-file">
                            <ImageIcon size={19} />
                            <div>
                                <strong>{file.name}</strong>
                                <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                            </div>
                        </div>
                    )}

                    {error && <p className="error-message">{error}</p>}

                    <button
                        type="button"
                        className="button-primary analyze-button"
                        disabled={!file || loading}
                        onClick={handleAnalyze}
                    >
                        {loading ? (
                            <>
                                <LoaderCircle className="spin" size={18} />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <ShieldCheck size={18} />
                                Analyze Image
                            </>
                        )}
                    </button>

                    <p className="privacy-note">
                        Your image is submitted to your configured SecureSight backend.
                    </p>
                </section>

                <section className="result-panel">
                    <div className="panel-heading">
                        <div>
                            <h2>Analysis Results</h2>
                            <p>Backend prediction will appear here.</p>
                        </div>
                        <ShieldCheck size={23} />
                    </div>

                    {!result ? (
                        <div className="empty-result">
                            <div className="empty-result-icon">
                                <ShieldCheck size={35} />
                            </div>
                            <h3>Awaiting image analysis</h3>
                            <p>
                                Select an image and click Analyze Image
                                to view the returned prediction.
                            </p>
                        </div>
                    ) : (



                        <div className="result-content redesigned-result">
                            <div className="result-status-icon">
                                <ShieldCheck size={34} />
                            </div>

                            <span className="small-label">FINAL PREDICTION</span>

                            <h2
                                className={
                                    result.prediction?.toLowerCase() === "fake"
                                        ? "result-fake"
                                        : "result-real"
                                }
                            >
                                {result.prediction}
                            </h2>

                            <div
                                className={`prediction-badge ${result.prediction?.toLowerCase() === "fake"
                                    ? "badge-fake"
                                    : "badge-real"
                                    }`}
                            >
                                <span className="status-dot" />
                                {result.prediction?.toLowerCase() === "fake"
                                    ? "AI-generated image detected"
                                    : "Real image detected"}
                            </div>

                            <div className="confidence-card">
                                <div className="confidence-heading">
                                    <div>
                                        <span className="confidence-label">
                                            Final Fusion Score (FAKE)
                                        </span>
                                        <h3>
                                            {(Number(result.fusion_score) * 100).toFixed(2)}%
                                        </h3>
                                    </div>

                                    <ShieldCheck size={27} className="confidence-icon" />
                                </div>

                                <div
                                    className="confidence-track"
                                    role="progressbar"
                                    aria-label="Final Fusion Score (FAKE)"
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-valuenow={Math.min(
                                        100,
                                        Math.max(0, Number(result.fusion_score) * 100)
                                    )}
                                >
                                    <div
                                        className={`confidence-fill ${result.prediction?.toLowerCase() === "fake"
                                            ? "confidence-fake"
                                            : "confidence-real"
                                            }`}
                                        style={{
                                            width: `${Math.min(
                                                100,
                                                Math.max(0, Number(result.fusion_score) * 100)
                                            )}%`,
                                        }}
                                    />
                                </div>

                                
                            </div>


                        </div>

                    )}
                </section>
            </div>
        </main>
    );
}
