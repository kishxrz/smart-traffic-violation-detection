'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid
} from 'recharts';
import {
  Car, AlertTriangle, Activity, Eye, Upload, RefreshCw,
  Shield, TrendingUp, Clock, Zap, CheckCircle, XCircle,
  AlertOctagon, Info, ChevronDown, ChevronUp, Play, Image as ImageIcon
} from 'lucide-react';
import {
  api, API_BASE,
  type AnalyticsSummary, type Violation,
  type VideoDetectionResult, type ImageDetectionResult
} from '@/lib/api';

// ── Color mapping ──────────────────────────────────────────────────────────────
const SEVERITY_COLOR: Record<string, string> = {
  LOW: '#06b6d4',
  MEDIUM: '#f59e0b',
  HIGH: '#ef4444',
  CRITICAL: '#a855f7',
};

const VIOLATION_COLOR: Record<string, string> = {
  NO_HELMET: '#f59e0b',
  RED_LIGHT: '#ef4444',
  WRONG_WAY: '#a855f7',
  LANE_VIOLATION: '#3b82f6',
};

const PIE_COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4'];

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({
  title,
  value,
  icon: Icon,
  color = '#3b82f6',
  subtitle,
}: {
  title: string;
  value: string | number;
  icon: React.ElementType;
  color?: string;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5 flex items-start gap-4 shadow-sm">
      <div
        className="flex items-center justify-center w-10 h-10 rounded-lg flex-shrink-0"
        style={{ backgroundColor: `${color}20` }}
      >
        <Icon className="w-5 h-5" style={{ color }} />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">{title}</p>
        <p className="text-2xl font-bold text-slate-100 mt-0.5">{value}</p>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

// ── Severity Badge ────────────────────────────────────────────────────────────
function SeverityBadge({
  severity,
  score,
  reasons,
}: {
  severity: string;
  score?: number;
  reasons?: string[];
}) {
  const color = SEVERITY_COLOR[severity] ?? '#94a3b8';
  return (
    <div className="group relative inline-block">
      <span
        className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-semibold cursor-help"
        style={{ backgroundColor: `${color}25`, color }}
      >
        {severity} {score !== undefined && <span className="opacity-75">({score} pts)</span>}
      </span>
      {reasons && reasons.length > 0 && (
        <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block z-50 w-64 bg-[#0a0e1a] border border-[#2a3550] text-slate-200 text-xs rounded-lg p-3 shadow-xl">
          <p className="font-semibold text-slate-100 mb-1 border-b border-[#2a3550] pb-1">Severity Factors:</p>
          <ul className="space-y-1 list-disc list-inside text-slate-300">
            {reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Upload Area ───────────────────────────────────────────────────────────────
function UploadZone({
  onFile,
  accept,
  label,
  loading,
  jobProgress,
}: {
  onFile: (f: File) => void;
  accept: string;
  label: string;
  loading: boolean;
  jobProgress?: { progress: number; status: string; frames: number } | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className="border-2 border-dashed border-[#2a3550] rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 transition-colors bg-[#111827]/40"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      {loading ? (
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
          <p className="text-sm font-medium text-blue-300">
            {jobProgress ? `Processing Video (${jobProgress.status} - ${jobProgress.progress}%)…` : 'Running Computer Vision & Violation Pipeline…'}
          </p>
          {jobProgress && (
            <div className="w-64 bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
              <div
                className="bg-blue-500 h-full transition-all duration-300 ease-out"
                style={{ width: `${jobProgress.progress}%` }}
              />
            </div>
          )}
          <p className="text-xs text-slate-500">
            {jobProgress ? `Frames Processed: ${jobProgress.frames}` : 'Detecting objects, tracking trajectories, and analyzing traffic rules'}
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-8 h-8 text-slate-400" />
          <p className="text-sm font-medium text-slate-200">{label}</p>
          <p className="text-xs text-slate-500">Click or drag & drop file to analyze</p>
        </div>
      )}
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'upload' | 'violations' | 'analytics'>('overview');
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [jobProgress, setJobProgress] = useState<{ progress: number; status: string; frames: number } | null>(null);
  
  // Video / Image analysis results state
  const [videoResult, setVideoResult] = useState<VideoDetectionResult | null>(null);
  const [imageResult, setImageResult] = useState<ImageDetectionResult | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showRawDetails, setShowRawDetails] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [sum, vio] = await Promise.all([
        api.analyticsSummary(),
        api.listViolations({ limit: 50 }),
      ]);
      setSummary(sum);
      setViolations(vio.violations);
      setIsOnline(true);
    } catch {
      setIsOnline(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleImageUpload = async (file: File) => {
    setLoading(true);
    setUploadError(null);
    setVideoResult(null);
    setImageResult(null);
    setJobProgress(null);
    try {
      const result = await api.detectImage(file);
      setImageResult(result);
      fetchData();
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Image upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (file: File) => {
    setLoading(true);
    setUploadError(null);
    setVideoResult(null);
    setImageResult(null);
    setJobProgress({ progress: 0, status: 'queued', frames: 0 });

    try {
      const submitRes = await api.detectVideo(file);
      const jobId = submitRes.job_id;

      // Poll background job status every 2 seconds
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const statusRes = await api.getVideoJobStatus(jobId);

        setJobProgress({
          progress: statusRes.progress || 0,
          status: statusRes.status,
          frames: statusRes.frames_processed || 0,
        });

        if (statusRes.status === 'completed') {
          if (statusRes.result) {
            setVideoResult(statusRes.result);
          }
          fetchData();
          break;
        } else if (statusRes.status === 'failed') {
          throw new Error(statusRes.error || 'Video background processing failed');
        }
      }
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Video processing failed');
    } finally {
      setLoading(false);
    }
  };

  // Derived chart data
  const violationTypeData = summary
    ? Object.entries(summary.violations_by_type).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))
    : [];

  const vehicleClassData = summary
    ? Object.entries(summary.vehicles_by_class).map(([k, v]) => ({ name: k, value: v }))
    : [];

  const timeSeriesData = summary?.traffic_volume_over_time?.slice(-60) ?? [];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'upload', label: 'Video & Image Analysis', icon: Upload },
    { id: 'violations', label: 'Violation Log', icon: AlertTriangle },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp },
  ] as const;

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-slate-100">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="border-b border-[#2a3550] bg-[#111827]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Eye className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-sm text-slate-100 tracking-wide">Smart Traffic Intelligence</span>
              <span className="ml-2 text-[10px] bg-blue-500/20 text-blue-400 font-mono border border-blue-500/30 rounded px-1.5 py-0.5">
                AI Vision v1.0
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs">
              <span
                className="w-2.5 h-2.5 rounded-full animate-pulse"
                style={{ backgroundColor: isOnline === null ? '#f59e0b' : isOnline ? '#22c55e' : '#ef4444' }}
              />
              <span className="text-slate-300 font-medium">
                {isOnline === null ? 'Connecting…' : isOnline ? 'Backend Online' : 'Backend Disconnected'}
              </span>
            </div>
            <button
              onClick={fetchData}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-[#2a3550] transition-colors"
              title="Refresh Data"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ── Offline Banner ───────────────────────────────────────────────── */}
      {isOnline === false && (
        <div className="bg-red-950/70 border-b border-red-800/50 px-4 py-2 text-center text-xs text-red-200 flex items-center justify-center gap-2">
          <AlertOctagon className="w-4 h-4 text-red-400" />
          <span>Cannot connect to FastAPI backend. Ensure uvicorn server is running on port 8000.</span>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* ── Tabs Navigation ─────────────────────────────────────────────── */}
        <nav className="flex gap-1 mb-6 border border-[#2a3550] rounded-xl p-1 bg-[#111827] w-fit shadow-inner">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === id
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#1a2235]'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </nav>

        {/* ── OVERVIEW TAB ────────────────────────────────────────────────── */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <StatCard
                title="OBJECT DETECTIONS"
                value={summary?.total_vehicles_detected ?? '—'}
                icon={Car}
                color="#3b82f6"
                subtitle="Total detections across frames"
              />
              <StatCard
                title="Unique Tracked"
                value={summary?.unique_vehicles_tracked ?? '—'}
                icon={Eye}
                color="#06b6d4"
                subtitle="Persistent IDs"
              />
              <StatCard
                title="Total Violations"
                value={summary?.total_violations ?? '—'}
                icon={AlertTriangle}
                color="#ef4444"
                subtitle="All rule breaches"
              />
              <StatCard
                title="No Helmet"
                value={summary?.violations_by_type?.NO_HELMET ?? 0}
                icon={Shield}
                color="#f59e0b"
              />
              <StatCard
                title="Red Light"
                value={summary?.violations_by_type?.RED_LIGHT ?? 0}
                icon={AlertOctagon}
                color="#ef4444"
              />
              <StatCard
                title="Wrong Way"
                value={summary?.violations_by_type?.WRONG_WAY ?? 0}
                icon={Zap}
                color="#a855f7"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
                  <span>Violations Breakdown by Type</span>
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                </h2>
                {violationTypeData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={violationTypeData} margin={{ left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                      <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #2a3550', borderRadius: 8 }}
                        labelStyle={{ color: '#f1f5f9' }}
                      />
                      <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyChart message="No violations recorded in active session." />
                )}
              </div>

              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
                  <span>Vehicle Class Distribution</span>
                  <Car className="w-4 h-4 text-blue-400" />
                </h2>
                {vehicleClassData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie
                        data={vehicleClassData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, percent }: { name?: string; percent?: number }) =>
                          name && percent !== undefined ? `${name} ${(percent * 100).toFixed(0)}%` : (name ?? '')
                        }
                        labelLine={false}
                      >
                        {vehicleClassData.map((_, i) => (
                          <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #2a3550', borderRadius: 8 }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyChart message="No vehicle detections registered." />
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard
                title="Avg Detection Confidence"
                value={summary ? `${(summary.avg_confidence * 100).toFixed(1)}%` : '—'}
                icon={CheckCircle}
                color="#22c55e"
                subtitle="YOLO inference confidence"
              />
              <StatCard
                title="Processing FPS"
                value={summary?.processing_fps?.toFixed(1) ?? '—'}
                icon={Activity}
                color="#3b82f6"
                subtitle="Pipeline throughput"
              />
              <StatCard
                title="Avg Latency"
                value={summary ? `${summary.avg_inference_time_ms.toFixed(0)}ms` : '—'}
                icon={Clock}
                color="#06b6d4"
                subtitle="Per-frame computation"
              />
            </div>
          </div>
        )}

        {/* ── VIDEO & IMAGE ANALYSIS TAB ──────────────────────────────────── */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div>
              <h1 className="text-lg font-bold text-slate-100">Traffic Camera Analysis</h1>
              <p className="text-sm text-slate-400 mt-0.5">
                Upload traffic video footage or snapshots to run object detection, Hungarian tracking, and rule-based violation reasoning.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <Play className="w-4 h-4 text-blue-400" />
                  Video Analysis (MP4, AVI)
                </h2>
                <DropZone
                  onFile={handleVideoUpload}
                  accept="video/mp4,video/avi,video/quicktime"
                  label="Upload traffic video file"
                  loading={loading}
                  jobProgress={jobProgress}
                />
              </div>

              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-cyan-400" />
                  Single Image Snapshot
                </h2>
                <DropZone
                  onFile={handleImageUpload}
                  accept="image/jpeg,image/png,image/webp"
                  label="Upload image snapshot"
                  loading={loading}
                />
              </div>
            </div>

            {uploadError && (
              <div className="rounded-xl bg-red-950/60 border border-red-800/50 p-4 flex items-start gap-3">
                <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-200">Analysis Failed</p>
                  <p className="text-xs text-red-300 mt-0.5">{uploadError}</p>
                </div>
              </div>
            )}

            {/* ── VIDEO RESULT DASHBOARD ────────────────────────────────────── */}
            {videoResult && (
              <div className="space-y-6">
                {/* 1. Annotated Output Video */}
                {videoResult.annotated_video_url && (
                  <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                    <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                      <Play className="w-4 h-4 text-green-400" />
                      Annotated Computer Vision Output Video
                    </h2>
                    <div className="relative rounded-lg overflow-hidden border border-[#2a3550] bg-black">
                      <video
                        controls
                        preload="metadata"
                        src={`${API_BASE}${videoResult.annotated_video_url}`}
                        className="w-full max-h-[480px] object-contain"
                      >
                        <div className="p-4 text-center text-xs text-red-400">
                          Unable to load annotated video. Check backend video generation/encoding.
                        </div>
                      </video>
                    </div>
                  </div>
                )}

                {/* 2. Analysis Summary Cards */}
                <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                  <h2 className="text-sm font-semibold text-slate-200 mb-4">Analysis Summary</h2>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                    <div className="bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-3">
                      <p className="text-[11px] text-slate-400 uppercase font-medium">Frames Processed</p>
                      <p className="text-xl font-bold text-slate-100 mt-1">{videoResult.frames_processed}</p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-3">
                      <p className="text-[11px] text-slate-400 uppercase font-medium">Object Detections</p>
                      <p className="text-xl font-bold text-blue-400 mt-1">{videoResult.summary.total_vehicles}</p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-3">
                      <p className="text-[11px] text-slate-400 uppercase font-medium">Unique Tracked</p>
                      <p className="text-xl font-bold text-cyan-400 mt-1">{videoResult.summary.unique_vehicles}</p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-3">
                      <p className="text-[11px] text-slate-400 uppercase font-medium">Violations Found</p>
                      <p className="text-xl font-bold text-red-400 mt-1">{videoResult.summary.total_violations}</p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-3">
                      <p className="text-[11px] text-slate-400 uppercase font-medium">Processing FPS</p>
                      <p className="text-xl font-bold text-green-400 mt-1">{videoResult.summary.processing_fps.toFixed(1)}</p>
                    </div>
                  </div>
                </div>

                {/* 3. Violation Breakdown Cards */}
                <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                  <h2 className="text-sm font-semibold text-slate-200 mb-3">Violation Breakdown</h2>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-[#0a0e1a] border border-amber-500/30 rounded-lg p-3">
                      <p className="text-xs text-amber-400 font-medium">No Helmet</p>
                      <p className="text-2xl font-bold text-slate-100 mt-1">
                        {videoResult.summary.violations_by_type?.NO_HELMET ?? 0}
                      </p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-purple-500/30 rounded-lg p-3">
                      <p className="text-xs text-purple-400 font-medium">Wrong Way</p>
                      <p className="text-2xl font-bold text-slate-100 mt-1">
                        {videoResult.summary.violations_by_type?.WRONG_WAY ?? 0}
                      </p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-red-500/30 rounded-lg p-3">
                      <p className="text-xs text-red-400 font-medium">Red Light</p>
                      <p className="text-2xl font-bold text-slate-100 mt-1">
                        {videoResult.summary.violations_by_type?.RED_LIGHT ?? 0}
                      </p>
                    </div>
                    <div className="bg-[#0a0e1a] border border-blue-500/30 rounded-lg p-3">
                      <p className="text-xs text-blue-400 font-medium">Lane Violation</p>
                      <p className="text-2xl font-bold text-slate-100 mt-1">
                        {videoResult.summary.violations_by_type?.LANE_VIOLATION ?? 0}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 4. Violation Results Table */}
                <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                  <h2 className="text-sm font-semibold text-slate-200 mb-3">Detected Violation Events ({videoResult.violations.length})</h2>
                  {videoResult.violations.length === 0 ? (
                    <div className="text-center py-8 text-slate-500 text-sm bg-[#0a0e1a] rounded-lg border border-[#2a3550]">
                      <CheckCircle className="w-6 h-6 text-green-400 mx-auto mb-2" />
                      No traffic violations detected in this video file.
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-[#2a3550]">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-[#0a0e1a] text-slate-400 font-semibold border-b border-[#2a3550]">
                          <tr>
                            <th className="px-3 py-2.5">Time</th>
                            <th className="px-3 py-2.5">Vehicle</th>
                            <th className="px-3 py-2.5">Violation</th>
                            <th className="px-3 py-2.5">Confidences</th>
                            <th className="px-3 py-2.5">Severity</th>
                            <th className="px-3 py-2.5">Evidence</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#2a3550]">
                          {videoResult.violations.map((v) => (
                            <tr key={v.id} className="hover:bg-[#2a3550]/30 transition-colors">
                              <td className="px-3 py-2.5 font-mono text-slate-300">
                                {v.timestamp.toFixed(2)}s <span className="text-slate-500">(f#{v.frame_number})</span>
                              </td>
                              <td className="px-3 py-2.5 font-medium text-slate-200">
                                #{v.vehicle_id} <span className="text-slate-400 capitalize">({v.vehicle_class})</span>
                              </td>
                              <td className="px-3 py-2.5">
                                <span
                                  className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold"
                                  style={{
                                    backgroundColor: `${VIOLATION_COLOR[v.violation_type] ?? '#3b82f6'}20`,
                                    color: VIOLATION_COLOR[v.violation_type] ?? '#3b82f6',
                                  }}
                                >
                                  {v.violation_type.replace(/_/g, ' ')}
                                </span>
                              </td>
                              <td className="px-3 py-2.5 text-slate-300">
                                <div className="flex flex-col gap-0.5 text-[11px]">
                                  <span>Violation: <b>{((v.violation_confidence ?? v.confidence) * 100).toFixed(0)}%</b></span>
                                  {v.detection_confidence !== undefined && (
                                    <span className="text-slate-500">YOLO: {(v.detection_confidence * 100).toFixed(0)}%</span>
                                  )}
                                </div>
                              </td>
                              <td className="px-3 py-2.5">
                                <SeverityBadge
                                  severity={v.severity}
                                  score={v.severity_score}
                                  reasons={v.severity_reasons}
                                />
                              </td>
                              <td className="px-3 py-2.5">
                                {v.evidence_path ? (
                                  <button
                                    onClick={() => setSelectedEvidence(`${API_BASE}/static/evidence/${v.evidence_path}`)}
                                    className="px-2 py-1 bg-blue-600/20 text-blue-400 hover:bg-blue-600/40 border border-blue-500/30 rounded text-[11px] font-medium transition-colors"
                                  >
                                    View Image
                                  </button>
                                ) : (
                                  <span className="text-slate-600 text-[11px]">N/A</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* 5. Collapsible Technical Details Accordion */}
                <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-4">
                  <button
                    onClick={() => setShowRawDetails(!showRawDetails)}
                    className="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white"
                  >
                    <span>Technical Response Details (Raw JSON)</span>
                    {showRawDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showRawDetails && (
                    <pre className="mt-3 text-xs text-slate-300 bg-[#0a0e1a] border border-[#2a3550] rounded-lg p-4 overflow-auto max-h-96 font-mono">
                      {JSON.stringify(videoResult, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}

            {/* Image Result Display */}
            {imageResult && (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5 space-y-4">
                <h2 className="text-sm font-semibold text-slate-200">Image Detection Results</h2>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="bg-[#0a0e1a] p-3 rounded-lg border border-[#2a3550]">
                    <span className="text-slate-400">Inference Latency:</span>
                    <p className="text-base font-bold text-blue-400">{imageResult.inference_time_ms.toFixed(1)} ms</p>
                  </div>
                  <div className="bg-[#0a0e1a] p-3 rounded-lg border border-[#2a3550]">
                    <span className="text-slate-400">Total Detections:</span>
                    <p className="text-base font-bold text-slate-100">{imageResult.summary.total_detections}</p>
                  </div>
                  <div className="bg-[#0a0e1a] p-3 rounded-lg border border-[#2a3550]">
                    <span className="text-slate-400">Violations Identified:</span>
                    <p className="text-base font-bold text-red-400">{imageResult.summary.total_violations}</p>
                  </div>
                  <div className="bg-[#0a0e1a] p-3 rounded-lg border border-[#2a3550]">
                    <span className="text-slate-400">Image Dimensions:</span>
                    <p className="text-base font-bold text-slate-300">{imageResult.frame_size.width}x{imageResult.frame_size.height}</p>
                  </div>
                </div>

                <div className="rounded-lg border border-[#2a3550] p-4 bg-[#0a0e1a]">
                  <button
                    onClick={() => setShowRawDetails(!showRawDetails)}
                    className="w-full flex items-center justify-between text-xs font-semibold text-slate-300"
                  >
                    <span>Technical Response Details (Raw JSON)</span>
                    {showRawDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                  {showRawDetails && (
                    <pre className="mt-3 text-xs text-slate-300 bg-[#111827] rounded-lg p-3 overflow-auto max-h-80 font-mono">
                      {JSON.stringify(imageResult, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── VIOLATIONS LOG TAB ───────────────────────────────────────────── */}
        {activeTab === 'violations' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h1 className="text-lg font-bold text-slate-100">Global Violation Records</h1>
              <span className="text-xs text-slate-400 bg-[#1a2235] border border-[#2a3550] rounded px-3 py-1 font-mono">
                {violations.length} total records
              </span>
            </div>

            {violations.length === 0 ? (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-12 text-center">
                <Info className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <p className="text-sm text-slate-400">
                  No violations logged yet. Upload a video file in the Analysis tab to begin.
                </p>
              </div>
            ) : (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] overflow-hidden">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[#111827] border-b border-[#2a3550] text-slate-400 uppercase font-semibold">
                    <tr>
                      <th className="px-4 py-3">Timestamp</th>
                      <th className="px-4 py-3">Vehicle</th>
                      <th className="px-4 py-3">Violation Type</th>
                      <th className="px-4 py-3">Violation Conf</th>
                      <th className="px-4 py-3">Severity & Factors</th>
                      <th className="px-4 py-3">Frame</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#2a3550]">
                    {violations.map((v) => (
                      <tr key={v.id} className="hover:bg-[#2a3550]/20 transition-colors">
                        <td className="px-4 py-3 font-mono text-slate-300">
                          {new Date(v.detected_at).toLocaleTimeString()}
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-slate-100 font-semibold">#{v.vehicle_id}</span>
                          <span className="text-slate-400 ml-1 text-xs capitalize">({v.vehicle_class})</span>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold"
                            style={{
                              backgroundColor: `${VIOLATION_COLOR[v.violation_type] ?? '#3b82f6'}20`,
                              color: VIOLATION_COLOR[v.violation_type] ?? '#3b82f6',
                            }}
                          >
                            {v.violation_type.replace(/_/g, ' ')}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-200 font-medium">
                          {((v.violation_confidence ?? v.confidence) * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3">
                          <SeverityBadge
                            severity={v.severity}
                            score={v.severity_score}
                            reasons={v.severity_reasons}
                          />
                        </td>
                        <td className="px-4 py-3 text-slate-500 font-mono">
                          #{v.frame_number}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── ANALYTICS TAB ───────────────────────────────────────────────── */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <h1 className="text-lg font-bold text-slate-100">System Analytics & Volume Trends</h1>

            <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
              <h2 className="text-sm font-semibold text-slate-200 mb-4">Real-Time Traffic Volume (Vehicles / Second)</h2>
              {timeSeriesData.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={timeSeriesData} margin={{ left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                    <XAxis
                      dataKey="time_seconds"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      tickFormatter={(v) => `${v.toFixed(0)}s`}
                    />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #2a3550', borderRadius: 8 }}
                      labelStyle={{ color: '#f1f5f9' }}
                      labelFormatter={(v) => `Time: ${Number(v).toFixed(1)}s`}
                    />
                    <Line
                      type="monotone"
                      dataKey="count"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Active Vehicles"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Process video footage to view traffic volume time-series." />
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Evidence Image Modal ────────────────────────────────────────────── */}
      {selectedEvidence && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-sm"
          onClick={() => setSelectedEvidence(null)}
        >
          <div className="bg-[#1a2235] border border-[#2a3550] rounded-xl max-w-2xl w-full p-4 space-y-3" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-[#2a3550] pb-2">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-blue-400" />
                Violation Evidence Image Frame
              </h3>
              <button
                onClick={() => setSelectedEvidence(null)}
                className="text-slate-400 hover:text-white p-1 rounded"
              >
                ✕
              </button>
            </div>
            <div className="overflow-hidden rounded-lg border border-[#2a3550] bg-black">
              {/* eslint-disable-next-html-next-line @next/next/no-img-element */}
              <img src={selectedEvidence} alt="Evidence snapshot" className="w-full object-contain max-h-[450px]" />
            </div>
            <div className="text-right">
              <button
                onClick={() => setSelectedEvidence(null)}
                className="px-4 py-1.5 bg-[#2a3550] text-slate-200 hover:bg-[#3a4768] rounded text-xs font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
      <div className="text-center">
        <Info className="w-6 h-6 mx-auto mb-2 opacity-50" />
        <p>{message}</p>
      </div>
    </div>
  );
}
