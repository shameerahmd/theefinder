"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import type { MapDetection } from "../components/ChennaiMap";

const ChennaiMap = dynamic(() => import("../components/ChennaiMap"), {
  ssr: false,
});

interface Detection extends MapDetection {
  detection_id?: string;
  explanation?: string;
  explanations?: string[];

  key_features?: {
    distance_to_industry_m?: number | null;
    industrial_feature_count_5km?: number | null;
    industrial_proximity_score?: number | null;

    tree_cover_pct?: number | null;
    cropland_pct?: number | null;
    built_up_pct?: number | null;

    detections_30d?: number | null;
    active_days_30d?: number | null;
    persistence_score?: number | null;
  };
}

type DashboardMode = "live" | "history";

interface ChennaiResponse {
  firms_detection_count?: number;
  processed_count?: number;
  classified_count?: number;
  failed_count?: number;

  detections?: Detection[];
  results?: Detection[];
  classifications?: Detection[];

  failures?: unknown[];
}

function prettyLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function confidencePercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function valueOrNA(value: number | null | undefined, suffix = ""): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${value.toFixed(2)}${suffix}`;
}

export default function Home() {
  const [mode, setMode] = useState<DashboardMode>("live");

  const [days, setDays] = useState<number>(5);

  const [historyDays, setHistoryDays] = useState<number>(30);

  const [historyClassification, setHistoryClassification] =
    useState<string>("ALL");

  const [detections, setDetections] = useState<Detection[]>([]);

  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const [loading, setLoading] = useState<boolean>(true);

  const [error, setError] = useState<string | null>(null);

  const [failedCount, setFailedCount] = useState<number>(0);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const endpoint =
        mode === "live"
          ? `/api/chennai?days=${days}&max_detections=25`
          : `/api/history?days=${historyDays}&limit=200${
              historyClassification === "ALL"
                ? ""
                : `&classification=${encodeURIComponent(
                    historyClassification,
                  )}`
            }`;

      const response = await fetch(endpoint, {
        cache: "no-store",
      });

      if (!response.ok) {
        const text = await response.text();

        throw new Error(
          text || `Request failed with status ${response.status}`,
        );
      }

      const payload: ChennaiResponse = await response.json();

      const rawDetections =
        payload.detections ??
        payload.results ??
        payload.classifications ??
        [];

      const normalized = rawDetections.map((detection) => ({
        ...detection,

        data_quality: detection.data_quality ?? {
          industrial_context_ok: false,
          landcover_ok: false,
        },
      }));

      setDetections(normalized);

      setFailedCount(
        mode === "live"
          ? payload.failed_count ?? payload.failures?.length ?? 0
          : 0,
      );

      if (normalized.length === 0) {
        setSelectedIndex(null);
      } else {
        const industrialIndex = normalized.findIndex(
          (item) =>
            item.final_classification ===
              "POTENTIAL_INDUSTRIAL_FIRE" ||
            item.final_classification ===
              "PERSISTENT_INDUSTRIAL_SOURCE",
        );

        setSelectedIndex(
          industrialIndex >= 0
            ? industrialIndex
            : 0,
        );
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : mode === "live"
            ? "Unable to load Chennai classification data."
            : "Unable to load historical PostGIS data.";

      setError(message);
      setDetections([]);
      setSelectedIndex(null);
    } finally {
      setLoading(false);
    }
  }, [
    mode,
    days,
    historyDays,
    historyClassification,
  ]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const handleSelect = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const selectedDetection =
    selectedIndex !== null ? (detections[selectedIndex] ?? null) : null;

  const counts = useMemo(() => {
    const output = {
      potentialIndustrial: 0,
      persistentIndustrial: 0,
      forest: 0,
      agricultural: 0,
      other: 0,
    };

    detections.forEach((detection) => {
      switch (detection.final_classification) {
        case "POTENTIAL_INDUSTRIAL_FIRE":
          output.potentialIndustrial += 1;
          break;

        case "PERSISTENT_INDUSTRIAL_SOURCE":
          output.persistentIndustrial += 1;
          break;

        case "FOREST_FIRE":
          output.forest += 1;
          break;

        case "AGRICULTURAL_OPEN_BURN":
          output.agricultural += 1;
          break;

        default:
          output.other += 1;
      }
    });

    return output;
  }, [detections]);

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-[1600px] px-6 py-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-red-600 text-xl font-bold text-white">
                  TF
                </div>

                <div>
                  <h1 className="text-2xl font-bold tracking-tight">
                    TheeFinder
                  </h1>

                  <p className="text-sm text-slate-500">
                    AI-Based Industrial Fire & Thermal Anomaly Detection
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="rounded-lg bg-emerald-50 px-3 py-2 text-sm font-medium text-emerald-700">
                ● System Online
              </div>

              <select
                value={days}
                onChange={(event) => setDays(Number(event.target.value))}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              >
                <option value={1}>Last 1 day</option>

                <option value={2}>Last 2 days</option>

                <option value={3}>Last 3 days</option>

                <option value={5}>Last 5 days</option>
              </select>

              <button
                type="button"
                onClick={() => void loadData()}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] space-y-6 px-6 py-6">
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <SummaryCard
            title="FIRMS Detections"
            value={detections.length}
            description={`${days}-day Chennai scan`}
          />

          <SummaryCard
            title="Potential Industrial"
            value={counts.potentialIndustrial}
            description="Requires verification"
          />

          <SummaryCard
            title="Forest"
            value={counts.forest}
            description="Model classification"
          />

          <SummaryCard
            title="Agricultural"
            value={counts.agricultural}
            description="Open-burn classification"
          />

          <SummaryCard
            title="Persistent Source"
            value={counts.persistentIndustrial}
            description="Recurring thermal source"
          />
        </section>

        {loading && (
          <div className="rounded-xl border border-blue-200 bg-blue-50 px-5 py-4 text-sm text-blue-800">
            Fetching FIRMS detections, geospatial context and AI
            classifications...
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-5">
            <div className="font-semibold text-red-800">
              Unable to load Chennai data
            </div>

            <p className="mt-2 text-sm text-red-700">{error}</p>

            <p className="mt-2 text-xs text-red-600">
              Confirm that the FastAPI backend is running on
              http://127.0.0.1:8000.
            </p>
          </div>
        )}

        <section className="grid gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
              <div>
                <h2 className="font-semibold">Chennai Thermal Activity Map</h2>

                <p className="text-xs text-slate-500">
                  FIRMS detections with TheeFinder AI classification
                </p>
              </div>

              <div className="text-xs text-slate-500">
                {detections.length} plotted
              </div>
            </div>

            <ChennaiMap
              detections={detections}
              selectedIndex={selectedIndex}
              onSelect={handleSelect}
            />
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-5 py-4">
              <h2 className="font-semibold">Detection Details</h2>
            </div>

            {selectedDetection ? (
              <DetectionDetails detection={selectedDetection} />
            ) : (
              <div className="p-6 text-sm text-slate-500">
                Select a thermal detection from the map or table.
              </div>
            )}
          </div>
        </section>

        <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
            <div>
              <h2 className="font-semibold">Detection Feed</h2>

              <p className="text-xs text-slate-500">
                Live classification results for the selected FIRMS window
              </p>
            </div>

            {failedCount > 0 && (
              <div className="rounded-md bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700">
                {failedCount} processing failure(s)
              </div>
            )}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[950px] text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-4 py-3">ID</th>

                  <th className="px-4 py-3">Location</th>

                  <th className="px-4 py-3">FRP</th>

                  <th className="px-4 py-3">Stage A</th>

                  <th className="px-4 py-3">Confidence</th>

                  <th className="px-4 py-3">Final Classification</th>

                  <th className="px-4 py-3">Acquisition</th>
                </tr>
              </thead>

              <tbody>
                {detections.map((detection, index) => (
                  <tr
                    key={
                      detection.detection_id ??
                      `${detection.latitude}-${detection.longitude}-${index}`
                    }
                    onClick={() => setSelectedIndex(index)}
                    className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                      selectedIndex === index ? "bg-blue-50" : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-medium">
                      {detection.detection_id ??
                        `TF-${String(index + 1).padStart(3, "0")}`}
                    </td>

                    <td className="px-4 py-3">
                      {detection.latitude.toFixed(5)},{" "}
                      {detection.longitude.toFixed(5)}
                    </td>

                    <td className="px-4 py-3">{detection.frp.toFixed(2)} MW</td>

                    <td className="px-4 py-3">
                      {prettyLabel(detection.stage_a_prediction)}
                    </td>

                    <td className="px-4 py-3">
                      {confidencePercent(detection.stage_a_confidence)}
                    </td>

                    <td className="px-4 py-3">
                      <ClassificationBadge
                        value={detection.final_classification}
                      />
                    </td>

                    <td className="px-4 py-3 text-xs text-slate-500">
                      {detection.acquisition_utc ?? "N/A"}
                    </td>
                  </tr>
                ))}

                {!loading && detections.length === 0 && (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-4 py-10 text-center text-slate-500"
                    >
                      No detections available.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">
          <strong>Prototype safety note:</strong> TheeFinder classifications are
          AI-assisted screening outputs. A label such as{" "}
          <strong>Potential Industrial Fire</strong> is not confirmation of an
          industrial accident. Satellite, infrastructure and field verification
          are required.
        </section>
      </div>
    </main>
  );
}

function SummaryCard({
  title,
  value,
  description,
}: {
  title: string;
  value: number;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="text-sm font-medium text-slate-500">{title}</div>

      <div className="mt-2 text-3xl font-bold">{value}</div>

      <div className="mt-1 text-xs text-slate-400">{description}</div>
    </div>
  );
}

function ClassificationBadge({ value }: { value: string }) {
  let className = "bg-slate-100 text-slate-700";

  if (value === "POTENTIAL_INDUSTRIAL_FIRE") {
    className = "bg-red-100 text-red-700";
  } else if (value === "PERSISTENT_INDUSTRIAL_SOURCE") {
    className = "bg-purple-100 text-purple-700";
  } else if (value === "FOREST_FIRE") {
    className = "bg-green-100 text-green-700";
  } else if (value === "AGRICULTURAL_OPEN_BURN") {
    className = "bg-amber-100 text-amber-700";
  }

  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${className}`}
    >
      {prettyLabel(value)}
    </span>
  );
}

function DetectionDetails({ detection }: { detection: Detection }) {
  const features = detection.key_features;

  const explanations =
    detection.explanations ??
    (detection.explanation ? [detection.explanation] : []);

  return (
    <div className="space-y-5 p-5">
      <ClassificationBadge value={detection.final_classification} />

      <div className="grid grid-cols-2 gap-3">
        <DetailItem label="Latitude" value={detection.latitude.toFixed(5)} />

        <DetailItem label="Longitude" value={detection.longitude.toFixed(5)} />

        <DetailItem label="FRP" value={`${detection.frp.toFixed(2)} MW`} />

        <DetailItem
          label="Stage A"
          value={prettyLabel(detection.stage_a_prediction)}
        />

        <DetailItem
          label="Stage A Confidence"
          value={confidencePercent(detection.stage_a_confidence)}
        />

        <DetailItem
          label="Stage B"
          value={
            detection.stage_b_prediction
              ? prettyLabel(detection.stage_b_prediction)
              : "Not applicable"
          }
        />

        <DetailItem
          label="Stage B Confidence"
          value={confidencePercent(detection.stage_b_confidence)}
        />

        <DetailItem
          label="Acquired"
          value={detection.acquisition_utc ?? "N/A"}
        />
      </div>

      {features && (
        <div>
          <h3 className="mb-3 text-sm font-semibold">Geospatial Features</h3>

          <div className="space-y-2 text-sm">
            <FeatureRow
              label="Distance to Industry"
              value={
                features.distance_to_industry_m === null ||
                features.distance_to_industry_m === undefined
                  ? "N/A"
                  : `${features.distance_to_industry_m.toFixed(0)} m`
              }
            />

            <FeatureRow
              label="Industrial Features ≤5 km"
              value={features.industrial_feature_count_5km?.toString() ?? "N/A"}
            />

            <FeatureRow
              label="Built-up"
              value={valueOrNA(features.built_up_pct, "%")}
            />

            <FeatureRow
              label="Cropland"
              value={valueOrNA(features.cropland_pct, "%")}
            />

            <FeatureRow
              label="Tree Cover"
              value={valueOrNA(features.tree_cover_pct, "%")}
            />

            <FeatureRow
              label="30-Day Detections"
              value={features.detections_30d?.toString() ?? "N/A"}
            />

            <FeatureRow
              label="Persistence Score"
              value={valueOrNA(features.persistence_score)}
            />
          </div>
        </div>
      )}

      <div>
        <h3 className="mb-3 text-sm font-semibold">Data Quality</h3>

        <div className="space-y-2">
          <QualityRow
            label="Industrial Context"
            ok={detection.data_quality.industrial_context_ok}
          />

          <QualityRow
            label="Land Cover"
            ok={detection.data_quality.landcover_ok}
          />
        </div>
      </div>

      {explanations.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold">Context Explanation</h3>

          <div className="space-y-2">
                        <ul className="space-y-2">
              {explanations.map((explanation, index) => (
                <li
                  key={index}
                  className="flex gap-2 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600"
                >
                  <span className="font-bold text-slate-400">•</span>
                  <span>{explanation}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="text-xs text-slate-400">{label}</div>

      <div className="mt-1 break-words text-sm font-medium">{value}</div>
    </div>
  );
}

function FeatureRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-slate-100 pb-2">
      <span className="text-slate-500">{label}</span>

      <span className="font-medium">{value}</span>
    </div>
  );
}

function QualityRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-sm">
      <span>{label}</span>

      <span
        className={
          ok ? "font-semibold text-emerald-600" : "font-semibold text-amber-600"
        }
      >
        {ok ? "Available" : "Unavailable"}
      </span>
    </div>
  );
}




