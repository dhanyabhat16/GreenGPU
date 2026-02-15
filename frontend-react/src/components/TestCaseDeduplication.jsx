import './TestCaseDeduplication.css'

export default function TestCaseDeduplication({ dedupData, loading, onRun }) {
  const res = dedupData?.result
  const original = res?.original_size ?? 0
  const reduced = res?.reduced_size ?? 0
  const removed = original - reduced
  const reductionPct = res?.reduction_percentage != null ? (res.reduction_percentage * 100).toFixed(1) : '0'
  const testCases = res?.test_cases ?? []

  return (
    <section className="card test-dedup">
      <h2 className="card__title">TEST CASE DEDUPLICATION</h2>
      <div className="dedup-summary">
        <span className="dedup-removed">{removed} Removed</span>
        <span className="dedup-pill">{reductionPct}% reduced</span>
      </div>

      <div className="dedup-breakdown">
        <div className="breakdown-item">
          <span className="breakdown-value gray">{original}</span>
          <span className="breakdown-label">ORIGINAL</span>
        </div>
        <div className="breakdown-item">
          <span className="breakdown-value green">{reduced}</span>
          <span className="breakdown-label">RETAINED</span>
        </div>
        <div className="breakdown-item">
          <span className="breakdown-value red">{removed}</span>
          <span className="breakdown-label">REMOVED</span>
        </div>
      </div>

      <div className="test-case-list">
        {testCases.length > 0 ? (
          testCases.map((tc) => (
            <div key={tc.id} className={`test-case-item ${tc.status}`}>
              <span className={`tc-dot ${tc.status}`} />
              <div className="tc-content">
                <span className="tc-id">{tc.id}</span>
                <span className="tc-label">{tc.label}</span>
                <span className={`tc-status ${tc.status}`}>
                  {tc.status === 'unique' ? 'unique ✓' : `${tc.similarity ?? 100}% sim → ${tc.similarTo ?? 'TC-001'}`}
                </span>
              </div>
            </div>
          ))
        ) : (
          <p className="tc-placeholder">Run deduplication to see test cases</p>
        )}
      </div>

      <button
        className="btn-run"
        onClick={onRun}
        disabled={loading}
      >
        {loading ? 'Running...' : 'Run Deduplicate'}
      </button>
    </section>
  )
}
