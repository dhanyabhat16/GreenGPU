import './ComputeDecision.css'

export default function ComputeDecision({ profileData, gpuStatus, loading, onRun }) {
  const stats = profileData?.stats
  const recommendGpu = !profileData?.recommend_cpu
  const deviceUsed = profileData?.device_used ?? 'gpu'
  const gpuUtil = stats?.gpu_utilization ?? 0
  const cpuUtil = stats?.cpu_utilization ?? 0
  const workloadScore = profileData?.workload_score ?? 65
  const threshold = 55

  return (
    <section className="card compute-decision">
      <h2 className="card__title">COMPUTE DECISION</h2>
      <div className="compute-recommendation">
        <span className={`recommendation-text ${recommendGpu ? 'gpu' : 'cpu'}`}>
          {recommendGpu ? 'GPU Recommended' : 'CPU Recommended'}
        </span>
        <span className={`device-pill ${deviceUsed}`}>
          ⚡ {deviceUsed.toUpperCase()}
        </span>
      </div>

      <div className="workload-score">
        <label>WORKLOAD SCORE</label>
        <div className="score-bar-container">
          <div className="score-bar">
            <div
              className="score-bar-fill"
              style={{ width: `${Math.min(100, (workloadScore / 125) * 100)}%` }}
            />
            <div
              className="score-threshold"
              style={{ left: `${(threshold / 125) * 100}%` }}
            />
          </div>
          <div className="score-labels">
            <span>CPU Zone</span>
            <span>{workloadScore}/125</span>
            <span>GPU Zone</span>
          </div>
          <span className="threshold-label">55 threshold</span>
        </div>
      </div>

      <div className="gauges">
        <div className="gauge" data-color="red">
          <div className="gauge-ring" style={{ '--pct': gpuUtil }}>
            <span className="gauge-value">{gpuUtil.toFixed(0)}%</span>
          </div>
          <label>GPU UTIL</label>
        </div>
        <div className="gauge" data-color="green">
          <div className="gauge-ring" style={{ '--pct': cpuUtil }}>
            <span className="gauge-value">{cpuUtil.toFixed(0)}%</span>
          </div>
          <label>CPU LOAD</label>
        </div>
        <div className="gauge" data-color="yellow">
          <div className="gauge-ring" style={{ '--pct': Math.min(100, workloadScore) }}>
            <span className="gauge-value">{Math.min(100, workloadScore)}%</span>
          </div>
          <label>SCORE</label>
        </div>
      </div>

      <button
        className="btn-run"
        onClick={onRun}
        disabled={loading}
      >
        {loading ? 'Running...' : 'Run Profile'}
      </button>
    </section>
  )
}
