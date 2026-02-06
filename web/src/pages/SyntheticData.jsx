import { useInspection } from "../context/InspectionContext.jsx";

export default function SyntheticData() {
  const { inspection } = useInspection();
  const images = inspection?.synthetic_images || [];

  return (
    <>
      <div>
        <div className="topbar-title">Synthetic Data</div>
        <div className="stat-foot">GAN-generated samples from the latest drift event</div>
      </div>

      <div className="card">
        <h3>Latest Synthetic Batch</h3>
        {images.length ? (
          <div className="grid-4">
            {images.map((img, idx) => (
              <div className="image-frame" key={`synth-${idx}`}>
                <img src={img} alt={`Synthetic ${idx + 1}`} />
              </div>
            ))}
          </div>
        ) : (
          <div className="stat-foot">No synthetic images generated yet.</div>
        )}
      </div>
    </>
  );
}
