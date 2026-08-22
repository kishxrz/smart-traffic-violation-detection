'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid, Legend
} from 'recharts';
import {
  Car, AlertTriangle, Activity, Eye, Upload, RefreshCw,
  Shield, TrendingUp, Clock, Zap, CheckCircle, XCircle,
  AlertOctagon, Info
} from 'lucide-react';
import { api, type AnalyticsSummary, type Violation } from '@/lib/api';

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
    <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5 flex items-start gap-4">
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
function SeverityBadge({ severity }: { severity: string }) {
  const color = SEVERITY_COLOR[severity] ?? '#94a3b8';
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
      style={{ backgroundColor: `${color}25`, color }}
    >
      {severity}
    </span>
  );
}

// ── Upload Area ───────────────────────────────────────────────────────────────
function UploadZone({
  onFile,
  accept,
  label,
  loading,
}: {
  onFile: (f: File) => void;
  accept: string;
  label: string;
  loading: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className="border-2 border-dashed border-[#2a3550] rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
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
        <div className="flex flex-col items-center gap-2">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
          <p className="text-sm text-slate-400">Processing…</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-8 h-8 text-slate-500" />
          <p className="text-sm font-medium text-slate-300">{label}</p>
          <p className="text-xs text-slate-500">Click or drag and drop</p>
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
  const [uploadResult, setUploadResult] = useState<Record<string, unknown> | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

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
    setUploadResult(null);
    try {
      const result = await api.detectImage(file);
      setUploadResult(result as unknown as Record<string, unknown>);
      fetchData();
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (file: File) => {
    setLoading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await api.detectVideo(file);
      setUploadResult(result as unknown as Record<string, unknown>);
      fetchData();
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  // Derived chart data
  const violationTypeData = summary
    ? Object.entries(summary.violations_by_type).map(([k, v]) => ({ name: k.replace('_', ' '), value: v }))
    : [];

  const vehicleClassData = summary
    ? Object.entries(summary.vehicles_by_class).map(([k, v]) => ({ name: k, value: v }))
    : [];

  const timeSeriesData = summary?.traffic_volume_over_time?.slice(-60) ?? [];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'upload', label: 'Analysis', icon: Upload },
    { id: 'violations', label: 'Violations', icon: AlertTriangle },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp },
  ] as const;

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-slate-100">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="border-b border-[#2a3550] bg-[#111827]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
              <Eye className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-sm text-slate-100">Smart Traffic AI</span>
            <span className="hidden sm:inline text-xs text-slate-500 border border-[#2a3550] rounded px-2 py-0.5">
              v1.0
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: isOnline === null ? '#f59e0b' : isOnline ? '#22c55e' : '#ef4444' }}
              />
              <span className="text-slate-400">
                {isOnline === null ? 'Connecting…' : isOnline ? 'Backend online' : 'Backend offline'}
              </span>
            </div>
            <button
              onClick={fetchData}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-[#2a3550] transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ── Offline Banner ───────────────────────────────────────────────── */}
      {isOnline === false && (
        <div className="bg-red-950/50 border-b border-red-800/50 px-4 py-2 text-center text-xs text-red-300">
          Cannot reach backend. Start the FastAPI server: <code className="font-mono">uvicorn app.main:app --reload</code>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* ── Tabs ─────────────────────────────────────────────────────────── */}
        <nav className="flex gap-1 mb-6 border border-[#2a3550] rounded-xl p-1 bg-[#111827] w-fit">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === id
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#1a2235]'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </nav>

        {/* ── Overview Tab ─────────────────────────────────────────────────── */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <StatCard
                title="Vehicles Detected"
                value={summary?.total_vehicles_detected ?? '—'}
                icon={Car}
                color="#3b82f6"
                subtitle="Total detections"
              />
              <StatCard
                title="Unique Vehicles"
                value={summary?.unique_vehicles_tracked ?? '—'}
                icon={Eye}
                color="#06b6d4"
                subtitle="Tracked IDs"
              />
              <StatCard
                title="Violations"
                value={summary?.total_violations ?? '—'}
                icon={AlertTriangle}
                color="#ef4444"
                subtitle="All types"
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

            {/* Charts row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Violations by type */}
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-300 mb-4">Violations by Type</h2>
                {violationTypeData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={violationTypeData} margin={{ left: -20 }}>
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
                  <EmptyChart message="No violations recorded yet." />
                )}
              </div>

              {/* Vehicle class distribution */}
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-300 mb-4">Vehicle Distribution</h2>
                {vehicleClassData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={200}>
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
                  <EmptyChart message="No detections yet." />
                )}
              </div>
            </div>

            {/* Performance row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard
                title="Avg Confidence"
                value={summary ? `${(summary.avg_confidence * 100).toFixed(1)}%` : '—'}
                icon={CheckCircle}
                color="#22c55e"
                subtitle="Detection confidence"
              />
              <StatCard
                title="Inference FPS"
                value={summary?.processing_fps?.toFixed(1) ?? '—'}
                icon={Activity}
                color="#3b82f6"
                subtitle="Frames per second"
              />
              <StatCard
                title="Avg Latency"
                value={summary ? `${summary.avg_inference_time_ms.toFixed(0)}ms` : '—'}
                icon={Clock}
                color="#06b6d4"
                subtitle="Per frame"
              />
            </div>
          </div>
        )}

        {/* ── Upload Tab ───────────────────────────────────────────────────── */}
        {activeTab === 'upload' && (
          <div className="space-y-6 max-w-3xl">
            <div>
              <h1 className="text-lg font-semibold text-slate-100">Video & Image Analysis</h1>
              <p className="text-sm text-slate-400 mt-1">
                Upload a traffic image or video to run the full detection pipeline.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-blue-400" />
                  Image Detection
                </h2>
                <UploadZone
                  onFile={handleImageUpload}
                  accept="image/jpeg,image/png,image/webp"
                  label="Upload an image (JPEG, PNG)"
                  loading={loading}
                />
              </div>

              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-400" />
                  Video Analysis
                </h2>
                <UploadZone
                  onFile={handleVideoUpload}
                  accept="video/mp4,video/avi,video/quicktime"
                  label="Upload a video (MP4, AVI)"
                  loading={loading}
                />
              </div>
            </div>

            {uploadError && (
              <div className="rounded-lg bg-red-950/50 border border-red-800/50 p-4 flex items-start gap-3">
                <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-300">{uploadError}</p>
              </div>
            )}

            {uploadResult && (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
                <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Analysis Results
                </h2>
                <pre className="text-xs text-slate-300 bg-[#0a0e1a] rounded-lg p-4 overflow-auto max-h-80">
                  {JSON.stringify(uploadResult, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* ── Violations Tab ───────────────────────────────────────────────── */}
        {activeTab === 'violations' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h1 className="text-lg font-semibold text-slate-100">Violation Log</h1>
              <span className="text-xs text-slate-400 bg-[#1a2235] border border-[#2a3550] rounded px-2 py-1">
                {violations.length} records
              </span>
            </div>

            {violations.length === 0 ? (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-12 text-center">
                <Info className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <p className="text-sm text-slate-400">
                  No violations recorded yet. Upload a video to start analysis.
                </p>
              </div>
            ) : (
              <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[#2a3550] text-left">
                      {['Time', 'Vehicle', 'Type', 'Confidence', 'Severity', 'Frame'].map((h) => (
                        <th key={h} className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {violations.map((v) => (
                      <tr
                        key={v.id}
                        className="border-b border-[#2a3550]/50 hover:bg-[#2a3550]/20 transition-colors"
                      >
                        <td className="px-4 py-3 text-slate-300 font-mono text-xs">
                          {new Date(v.detected_at).toLocaleTimeString()}
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-slate-100 font-medium">#{v.vehicle_id}</span>
                          <span className="text-slate-500 ml-1 text-xs">{v.vehicle_class}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                            style={{
                              backgroundColor: `${VIOLATION_COLOR[v.violation_type] ?? '#3b82f6'}20`,
                              color: VIOLATION_COLOR[v.violation_type] ?? '#3b82f6',
                            }}
                          >
                            {v.violation_type.replace(/_/g, ' ')}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-300">
                          {(v.confidence * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3">
                          <SeverityBadge severity={v.severity} />
                        </td>
                        <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                          {v.frame_number}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── Analytics Tab ────────────────────────────────────────────────── */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <h1 className="text-lg font-semibold text-slate-100">Analytics</h1>

            {/* Traffic volume over time */}
            <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
              <h2 className="text-sm font-semibold text-slate-300 mb-4">Traffic Volume Over Time</h2>
              {timeSeriesData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={timeSeriesData} margin={{ left: -20 }}>
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
                      name="Vehicles"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="Process a video to see traffic volume over time." />
              )}
            </div>

            {/* Severity breakdown */}
            <div className="rounded-xl border border-[#2a3550] bg-[#1a2235] p-5">
              <h2 className="text-sm font-semibold text-slate-300 mb-4">Violations by Severity</h2>
              {summary && Object.keys(summary.violations_by_severity).length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={Object.entries(summary.violations_by_severity).map(([k, v]) => ({
                      name: k,
                      count: v,
                    }))}
                    margin={{ left: -20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1a2235', border: '1px solid #2a3550', borderRadius: 8 }}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {Object.keys(summary.violations_by_severity).map((k, i) => (
                        <Cell key={i} fill={SEVERITY_COLOR[k] ?? '#3b82f6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart message="No violations recorded yet." />
              )}
            </div>
          </div>
        )}
      </div>
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
