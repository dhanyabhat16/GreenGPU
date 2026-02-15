import { useState, useEffect, useCallback } from 'react'
import { verify, profile, deduplicate, evaluate } from './api'
import MetricCards from './components/MetricCards'
import ComputeDecision from './components/ComputeDecision'
import TestCaseDeduplication from './components/TestCaseDeduplication'
import SustainabilityImpact from './components/SustainabilityImpact'
import './App.css'

function App() {
  const [gpuStatus, setGpuStatus] = useState(null)
  const [profileData, setProfileData] = useState(null)
  const [dedupData, setDedupData] = useState(null)
  const [evalData, setEvalData] = useState(null)
  const [loading, setLoading] = useState({ verify: true, profile: false, dedup: false, eval: false })
  const [error, setError] = useState(null)

  const runVerify = useCallback(async () => {
    setLoading(l => ({ ...l, verify: true }))
    setError(null)
    try {
      const d = await verify()
      setGpuStatus(d)
    } catch (e) {
      setError(e.message)
      setGpuStatus({ cuda_available: false })
    } finally {
      setLoading(l => ({ ...l, verify: false }))
    }
  }, [])

  const runProfile = useCallback(async () => {
    setLoading(l => ({ ...l, profile: true }))
    setError(null)
    try {
      const d = await profile()
      setProfileData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, profile: false }))
    }
  }, [])

  const runDedup = useCallback(async () => {
    setLoading(l => ({ ...l, dedup: true }))
    setError(null)
    try {
      const d = await deduplicate()
      setDedupData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, dedup: false }))
    }
  }, [])

  const runEval = useCallback(async () => {
    setLoading(l => ({ ...l, eval: true }))
    setError(null)
    try {
      const d = await evaluate()
      setEvalData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(l => ({ ...l, eval: false }))
    }
  }, [])

  useEffect(() => { runVerify() }, [runVerify])

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">◆</span>
            <span>GreenGPU Optimizer</span>
          </div>
          <span className="tagline">INTELLIGENT ML COMPUTE OPTIMIZATION</span>
        </div>
        <nav className="nav">
          <a href="#" className="nav-link active">Dashboard</a>
        </nav>
        <div className="header-right">
          <div className={`status-pill ${gpuStatus?.cuda_available ? 'online' : ''}`}>
            <span className="status-dot" />
            SYSTEM {gpuStatus?.cuda_available ? 'ONLINE' : loading.verify ? 'CHECKING...' : 'OFFLINE'}
          </div>
        </div>
      </header>

      <main className="main">
        {error && <div className="toast error">{error}</div>}

        <MetricCards
          profileData={profileData}
          dedupData={dedupData}
          evalData={evalData}
        />

        <div className="grid-2">
          <ComputeDecision
            profileData={profileData}
            gpuStatus={gpuStatus}
            loading={loading.profile}
            onRun={runProfile}
          />
          <TestCaseDeduplication
            dedupData={dedupData}
            loading={loading.dedup}
            onRun={runDedup}
          />
        </div>

        <SustainabilityImpact
          profileData={profileData}
          dedupData={dedupData}
          evalData={evalData}
        />
      </main>
    </div>
  )
}

export default App
