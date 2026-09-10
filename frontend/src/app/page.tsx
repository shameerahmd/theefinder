"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import type { MapDetection } from "../components/ChennaiMap";

const ChennaiMap = dynamic(() => import("../components/ChennaiMap"), {
  ssr: false,
});

interface Detection extends MapDetection {
  detection_id?: string;
  satellite?: string | null;
  source?: string | null;

  explanation?: string | string[];
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

    nearest_major_road_distance_m?: number | null;
    nearest_major_road_class?: string | null;
    nearest_major_road_name?: string | null;
    nearest_major_road_ref?: string | null;
    road_context_source?: string | null;

    nearest_industrial_type?: string | null;
    nearest_industrial_name?: string | null;
    nearest_industrial_facility_distance_m?: number | null;

    nearest_specialized_industrial_type?: string | null;
    nearest_specialized_industrial_name?: string | null;
    nearest_specialized_industrial_distance_m?: number | null;

    refinery_count?: number | null;
    power_plant_count?: number | null;
    steel_metal_plant_count?: number | null;
    factory_count?: number | null;
    mine_quarry_count?: number | null;
    flare_count?: number | null;
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

function prettyLabel(value: string | null | undefined): string {
  if (!value) {
    return "â€”";
  }

  if (value === "UNCERTAIN_INDUSTRIAL_EVENT") {
    return "Needs Review";
  }

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

function valueOrNA(
  value: number | null | undefined,
  suffix = "",
  decimals = 2,
): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${value.toFixed(decimals)}${suffix}`;
}

function formatAcquisitionTime(value: string | null | undefined): string {
  if (!value) {
    return "N/A";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "Asia/Kolkata",
    timeZoneName: "short",
  }).format(date);
}

function getAccent(value: string): {
  border: string;
  bg: string;
  text: string;
  dot: string;
  top: string;
} {
  switch (value) {
    case "INDUSTRIAL_THERMAL_ANOMALY":
      return {
        border: "border-red-200",
        bg: "bg-red-50",
        text: "text-red-700",
        dot: "bg-red-500",
        top: "border-t-red-500",
      };

    case "POTENTIAL_VEHICLE_FIRE":
      return {
        border: "border-orange-200",
        bg: "bg-orange-50",
        text: "text-orange-700",
        dot: "bg-orange-500",
        top: "border-t-orange-500",
      };

    case "PERSISTENT_INDUSTRIAL_SOURCE":
      return {
        border: "border-sky-200",
        bg: "bg-sky-50",
        text: "text-sky-700",
        dot: "bg-sky-500",
        top: "border-t-sky-500",
      };

    case "FOREST_FIRE":
      return {
        border: "border-emerald-200",
        bg: "bg-emerald-50",
        text: "text-emerald-700",
        dot: "bg-emerald-500",
        top: "border-t-emerald-500",
      };

    case "AGRICULTURAL_OPEN_BURN":
      return {
        border: "border-amber-200",
        bg: "bg-amber-50",
        text: "text-amber-700",
        dot: "bg-amber-500",
        top: "border-t-amber-500",
      };

    case "UNCERTAIN_INDUSTRIAL_EVENT":
      return {
        border: "border-violet-200",
        bg: "bg-violet-50",
        text: "text-violet-700",
        dot: "bg-violet-500",
        top: "border-t-violet-500",
      };

    default:
      return {
        border: "border-slate-200",
        bg: "bg-slate-50",
        text: "text-slate-700",
        dot: "bg-slate-500",
        top: "border-t-slate-500",
      };
  }
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
                : `&classification=${encodeURIComponent(historyClassification)}`
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
        payload.detections ?? payload.results ?? payload.classifications ?? [];

      const normalized = rawDetections.map((detection) => ({
        ...detection,

        data_quality: detection.data_quality ?? {
          industrial_context_ok: false,
          landcover_ok: false,
          persistence_status: "NOT_REQUIRED",
          persistence_ok: null,
          road_context_status: "NOT_REQUIRED",
          road_context_ok: null,
        },
      }));

      setDetections(normalized);

      setFailedCount(
        mode === "live"
          ? (payload.failed_count ?? payload.failures?.length ?? 0)
          : 0,
      );

      if (normalized.length === 0) {
        setSelectedIndex(null);
      } else {
        const priorityIndex = normalized.findIndex(
          (item) =>
            item.final_classification === "INDUSTRIAL_THERMAL_ANOMALY" ||
            item.final_classification === "POTENTIAL_VEHICLE_FIRE" ||
            item.final_classification === "UNCERTAIN_INDUSTRIAL_EVENT" ||
            item.final_classification === "PERSISTENT_INDUSTRIAL_SOURCE",
        );

        setSelectedIndex(priorityIndex >= 0 ? priorityIndex : 0);
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
  }, [mode, days, historyDays, historyClassification]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadData();
    }, 0);

    return () => {
      window.clearTimeout(timer);
    };
  }, [loadData]);

  const changeMode = (nextMode: DashboardMode) => {
    setSelectedIndex(null);
    setMode(nextMode);
  };

  const handleSelect = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const selectedDetection =
    selectedIndex !== null ? (detections[selectedIndex] ?? null) : null;

  const counts = useMemo(() => {
    const output = {
      potentialIndustrial: 0,
      potentialVehicle: 0,
      persistentIndustrial: 0,
      forest: 0,
      agricultural: 0,
      uncertain: 0,
    };

    detections.forEach((detection) => {
      switch (detection.final_classification) {
        case "INDUSTRIAL_THERMAL_ANOMALY":
          output.potentialIndustrial += 1;
          break;

        case "POTENTIAL_VEHICLE_FIRE":
          output.potentialVehicle += 1;
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

        case "UNCERTAIN_INDUSTRIAL_EVENT":
          output.uncertain += 1;
          break;
      }
    });

    return output;
  }, [detections]);

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white/95 shadow-sm backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-6 py-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-red-600 to-orange-500 text-xl font-black text-white shadow-sm">
                TF
                <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-white bg-emerald-500" />
              </div>

              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-2xl font-black tracking-tight">
                    Thee Finder {"\u{1F525}"}
                  </h1>

                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-black uppercase tracking-[0.12em] text-slate-500">
                    Fire Detection & Thermal Anomaly Classifier
                  </span>
                </div>

                <p className="text-sm text-slate-500">
                  Fire Detection & Thermal Anomaly Classifier
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                System Online
              </div>

              <div
                className={`rounded-lg border px-3 py-2 text-xs font-black uppercase tracking-wide ${
                  mode === "live"
                    ? "border-blue-200 bg-blue-50 text-blue-700"
                    : "border-violet-200 bg-violet-50 text-violet-700"
                }`}
              >
                {mode === "live" ? "Live Data" : "PostGIS History"}
              </div>

              <div className="flex rounded-lg border border-slate-300 bg-slate-100 p-1">
                <button
                  type="button"
                  onClick={() => changeMode("live")}
                  className={`rounded-md px-3 py-1.5 text-sm font-semibold transition ${
                    mode === "live"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  Live
                </button>

                <button
                  type="button"
                  onClick={() => changeMode("history")}
                  className={`rounded-md px-3 py-1.5 text-sm font-semibold transition ${
                    mode === "history"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  History
                </button>
              </div>

              {mode === "live" ? (
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
              ) : (
                <>
                  <select
                    value={historyDays}
                    onChange={(event) =>
                      setHistoryDays(Number(event.target.value))
                    }
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
                  >
                    <option value={7}>Last 7 days</option>
                    <option value={30}>Last 30 days</option>
                    <option value={90}>Last 90 days</option>
                  </select>

                  <select
                    value={historyClassification}
                    onChange={(event) =>
                      setHistoryClassification(event.target.value)
                    }
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
                  >
                    <option value="ALL">All classifications</option>
                    <option value="INDUSTRIAL_THERMAL_ANOMALY">
                      Industrial Thermal Anomaly
                    </option>
                    <option value="POTENTIAL_VEHICLE_FIRE">
                      Potential Vehicle Fire
                    </option>
                    <option value="PERSISTENT_INDUSTRIAL_SOURCE">
                      Persistent Industrial Source
                    </option>
                    <option value="FOREST_FIRE">Forest Fire</option>
                    <option value="AGRICULTURAL_OPEN_BURN">
                      Agricultural Open Burn
                    </option>
                    <option value="UNCERTAIN_INDUSTRIAL_EVENT">
                      Uncertain Industrial Event
                    </option>
                  </select>
                </>
              )}

              <button
                type="button"
                onClick={() => void loadData()}
                disabled={loading}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] space-y-6 px-6 py-6">
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6">
          <SummaryCard
            title={mode === "live" ? "FIRMS Detections" : "Stored Detections"}
            value={detections.length}
            description={
              mode === "live"
                ? `${days}-day Chennai live scan`
                : `${historyDays}-day PostGIS history`
            }
            accent="slate"
          />

          <SummaryCard
            title="Industrial Thermal Anomaly"
            value={counts.potentialIndustrial}
            description="Requires verification"
            accent="red"
          />

          <SummaryCard
            title="Potential Vehicle"
            value={counts.potentialVehicle}
            description="Contextual road-screening"
            accent="orange"
          />

          <SummaryCard
            title="Forest"
            value={counts.forest}
            description="Model classification"
            accent="green"
          />

          <SummaryCard
            title="Agricultural"
            value={counts.agricultural}
            description="Open-burn classification"
            accent="amber"
          />

          <SummaryCard
            title="Persistent Source"
            value={counts.persistentIndustrial}
            description="Recurring thermal source"
            accent="sky"
          />
        </section>

        {counts.uncertain > 0 && (
          <div className="rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-800">
            <strong>{counts.uncertain}</strong> uncertain industrial event(s)
            require additional review.
          </div>
        )}

        {loading && (
          <div className="rounded-xl border border-blue-200 bg-blue-50 px-5 py-4 text-sm text-blue-800">
            <span className="mr-2 inline-flex h-2 w-2 animate-pulse rounded-full bg-blue-500" />
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
              Confirm that the FastAPI backend and THEEFINDER_BACKEND_URL are
              configured correctly.
            </p>
          </div>
        )}

        <section className="grid items-start gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
          <div className="self-start overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
                  <h2 className="font-bold">Chennai Thermal Activity Map</h2>
                </div>

                <p className="mt-1 text-xs text-slate-500">
                  FIRMS detections with TheeFinder AI classification
                </p>
              </div>

              <MapLegend />
            </div>

            <ChennaiMap
              detections={detections}
              selectedIndex={selectedIndex}
              onSelect={handleSelect}
            />
          </div>

          <div className="self-start overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 bg-slate-50 px-5 py-4">
              <div className="text-[10px] font-black uppercase tracking-[0.12em] text-slate-400">
                Selected anomaly
              </div>
              <h2 className="mt-1 font-bold">Detection Details</h2>
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
              <h2 className="font-bold">Detection Feed</h2>

              <p className="text-xs text-slate-500">
                Classification results for the selected data window
              </p>
            </div>

            {failedCount > 0 && (
              <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700">
                {failedCount} processing failure(s)
              </div>
            )}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[1080px] text-left text-sm">
              <thead className="bg-slate-50 text-[10px] font-black uppercase tracking-[0.08em] text-slate-500">
                <tr>
                  <th className="px-4 py-3">ID</th>
                  <th className="px-4 py-3">Location</th>
                  <th className="px-4 py-3">FRP</th>
                  <th className="px-4 py-3">Stage A</th>
                  <th className="px-4 py-3">A Confidence</th>
                  <th className="px-4 py-3">Stage B</th>
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
                    className={`cursor-pointer border-t border-slate-100 transition hover:bg-slate-50 ${
                      selectedIndex === index
                        ? "bg-blue-50/70 ring-1 ring-inset ring-blue-100"
                        : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-bold">
                      {detection.detection_id ??
                        `TF-${String(index + 1).padStart(3, "0")}`}
                    </td>

                    <td className="px-4 py-3 font-mono text-xs text-slate-600">
                      {detection.latitude.toFixed(5)},{" "}
                      {detection.longitude.toFixed(5)}
                    </td>

                    <td className="px-4 py-3 font-semibold">
                      {detection.frp.toFixed(2)} MW
                    </td>

                    <td className="px-4 py-3">
                      {prettyLabel(detection.stage_a_prediction)}
                    </td>

                    <td className="px-4 py-3">
                      {confidencePercent(detection.stage_a_confidence)}
                    </td>

                    <td className="px-4 py-3">
                      {detection.stage_b_prediction
                        ? `${prettyLabel(
                            detection.stage_b_prediction,
                          )} (${confidencePercent(
                            detection.stage_b_confidence,
                          )})`
                        : "Not applicable"}
                    </td>

                    <td className="px-4 py-3">
                      <ClassificationBadge
                        value={detection.final_classification}
                      />
                    </td>

                    <td className="px-4 py-3 text-xs text-slate-500">
                      {formatAcquisitionTime(detection.acquisition_utc)}
                    </td>
                  </tr>
                ))}

                {!loading && detections.length === 0 && (
                  <tr>
                    <td
                      colSpan={8}
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

        <section className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm leading-6 text-amber-900">
          <strong>Prototype safety note:</strong> TheeFinder classifications are
          AI-assisted screening outputs. A label such as{" "}
          <strong>Industrial Thermal Anomaly</strong> or{" "}
          <strong>Potential Vehicle Fire</strong> is not confirmation of an
          incident. Satellite, infrastructure and field verification are
          required.
        </section>
      </div>
    </main>
  );
}

function SummaryCard({
  title,
  value,
  description,
  accent,
}: {
  title: string;
  value: number;
  description: string;
  accent: "slate" | "red" | "orange" | "green" | "amber" | "sky";
}) {
  const topBorder = {
    slate: "border-t-slate-500",
    red: "border-t-red-500",
    orange: "border-t-orange-500",
    green: "border-t-emerald-500",
    amber: "border-t-amber-500",
    sky: "border-t-sky-500",
  }[accent];

  return (
    <div
      className={`rounded-2xl border border-slate-200 border-t-[3px] ${topBorder} bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md`}
    >
      <div className="text-sm font-medium text-slate-500">{title}</div>
      <div className="mt-2 text-3xl font-black tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-slate-400">{description}</div>
    </div>
  );
}

function ClassificationBadge({ value }: { value: string }) {
  const accent = getAccent(value);

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${accent.border} ${accent.bg} ${accent.text}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${accent.dot}`} />
      {prettyLabel(value)}
    </span>
  );
}

function MapLegend() {
  const items = [
    ["bg-red-500", "Industrial"],
    ["bg-orange-500", "Vehicle"],
    ["bg-emerald-500", "Forest"],
    ["bg-amber-500", "Agricultural"],
    ["bg-sky-500", "Persistent"],
    ["bg-violet-500", "Uncertain"],
  ];

  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1.5">
      {items.map(([dot, label]) => (
        <span
          key={label}
          className="flex items-center gap-1.5 text-[10px] font-semibold text-slate-500"
        >
          <span className={`h-2 w-2 rounded-full ${dot}`} />
          {label}
        </span>
      ))}
    </div>
  );
}

function DetectionDetails({ detection }: { detection: Detection }) {
  const features = detection.key_features;

  const explanations =
    detection.explanations ??
    (Array.isArray(detection.explanation)
      ? detection.explanation
      : detection.explanation
        ? [detection.explanation]
        : []);

  const vehicle = detection.final_classification === "POTENTIAL_VEHICLE_FIRE";

  const accent = getAccent(detection.final_classification);

  return (
    <div className="space-y-5 p-5">
      <div className={`rounded-xl border p-4 ${accent.border} ${accent.bg}`}>
        <div className="text-[10px] font-black uppercase tracking-[0.12em] text-slate-400">
          Final classification
        </div>

        <div className="mt-2">
          <ClassificationBadge value={detection.final_classification} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <DetailItem label="Latitude" value={detection.latitude.toFixed(5)} />
        <DetailItem label="Longitude" value={detection.longitude.toFixed(5)} />
        <DetailItem label="FRP" value={`${detection.frp.toFixed(2)} MW`} />
        <DetailItem label="Satellite" value={detection.satellite ?? "VIIRS"} />

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

        <div className="col-span-2">
          <DetailItem
            label="Acquired"
            value={formatAcquisitionTime(detection.acquisition_utc)}
          />
        </div>
      </div>

      {features && (
        <div>
          <h3 className="mb-3 text-sm font-bold">
            Geospatial & Historical Context
          </h3>

          <div className="overflow-hidden rounded-xl border border-slate-200">
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
              label="Industrial Features within 5 km"
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
              label="Active Days / 30d"
              value={features.active_days_30d?.toString() ?? "N/A"}
            />

            <FeatureRow
              label="Persistence Score"
              value={valueOrNA(features.persistence_score)}
              last={!vehicle}
            />

            {vehicle && (
              <>
                <FeatureRow
                  label="Nearest Drivable Road"
                  value={
                    features.nearest_major_road_name ??
                    features.nearest_major_road_class ??
                    "N/A"
                  }
                />

                <FeatureRow
                  label="Road Distance"
                  value={
                    features.nearest_major_road_distance_m === null ||
                    features.nearest_major_road_distance_m === undefined
                      ? "N/A"
                      : `${features.nearest_major_road_distance_m.toFixed(0)} m`
                  }
                  last
                />
              </>
            )}
          </div>
        </div>
      )}
      {features &&
        (features.nearest_specialized_industrial_type ||
          features.nearest_industrial_type ||
          features.refinery_count ||
          features.power_plant_count ||
          features.steel_metal_plant_count ||
          features.factory_count ||
          features.mine_quarry_count ||
          features.flare_count) && (
          <div>
            <h3 className="mb-3 text-sm font-bold">
              Industrial Facility Context
            </h3>

            <div className="overflow-hidden rounded-xl border border-slate-200">
              <FeatureRow
                label="Nearest Facility Type"
                value={prettyLabel(
                  features.nearest_specialized_industrial_type ??
                    features.nearest_industrial_type ??
                    "UNKNOWN",
                )}
              />

              <FeatureRow
                label="Facility Name"
                value={
                  features.nearest_specialized_industrial_name ??
                  features.nearest_industrial_name ??
                  "Unnamed OSM facility"
                }
              />

              <FeatureRow
                label="Facility Distance"
                value={
                  features.nearest_specialized_industrial_distance_m !== null &&
                  features.nearest_specialized_industrial_distance_m !==
                    undefined
                    ? `${features.nearest_specialized_industrial_distance_m.toFixed(0)} m`
                    : features.nearest_industrial_facility_distance_m !==
                          null &&
                        features.nearest_industrial_facility_distance_m !==
                          undefined
                      ? `${features.nearest_industrial_facility_distance_m.toFixed(0)} m`
                      : "N/A"
                }
              />

              <FeatureRow
                label="Refinery OSM Matches"
                value={features.refinery_count?.toString() ?? "0"}
              />

              <FeatureRow
                label="Power Plant OSM Matches"
                value={features.power_plant_count?.toString() ?? "0"}
              />

              <FeatureRow
                label="Factory OSM Matches"
                value={features.factory_count?.toString() ?? "0"}
              />

              <FeatureRow
                label="Steel / Metal Plant OSM Matches"
                value={features.steel_metal_plant_count?.toString() ?? "0"}
              />

              <FeatureRow
                label="Mine / Quarry OSM Matches"
                value={features.mine_quarry_count?.toString() ?? "0"}
              />

              <FeatureRow
                label="Flare OSM Matches"
                value={features.flare_count?.toString() ?? "0"}
                last
              />
            </div>

            <p className="mt-2 text-[10px] leading-4 text-slate-400">
              Facility counts represent nearby OpenStreetMap feature matches and
              may not equal the number of distinct physical facilities.
            </p>
          </div>
        )}
      <div>
        <h3 className="mb-3 text-sm font-bold">Data Quality</h3>

        <div className="grid grid-cols-2 gap-2">
          <QualityRow
            label="Industrial Context"
            ok={detection.data_quality.industrial_context_ok}
          />

          <QualityRow
            label="Land Cover"
            ok={detection.data_quality.landcover_ok}
          />

          <QualityRow
            label="Persistence"
            ok={detection.data_quality.persistence_ok}
            neutral={detection.data_quality.persistence_ok === null}
          />

          <QualityRow
            label="Road Context"
            ok={detection.data_quality.road_context_ok}
            neutral={
              detection.data_quality.road_context_ok === null ||
              detection.data_quality.road_context_ok === undefined
            }
          />
        </div>
      </div>

      {explanations.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-bold">Context Explanation</h3>

          <ul className="space-y-2">
            {explanations.map((explanation, index) => (
              <li
                key={`${index}-${explanation}`}
                className="flex gap-2 rounded-lg border border-slate-100 bg-slate-50 p-3 text-xs leading-5 text-slate-600"
              >
                <span className="mt-0.5 font-black text-blue-500">
                  {"\u2022"}
                </span>
                <span>{explanation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="h-full rounded-lg border border-slate-100 bg-slate-50 p-3">
      <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
        {label}
      </div>

      <div className="mt-1 break-words text-sm font-semibold">{value}</div>
    </div>
  );
}

function FeatureRow({
  label,
  value,
  last = false,
}: {
  label: string;
  value: string;
  last?: boolean;
}) {
  return (
    <div
      className={`flex justify-between gap-4 bg-white px-3 py-2.5 text-sm ${
        last ? "" : "border-b border-slate-100"
      }`}
    >
      <span className="text-slate-500">{label}</span>
      <span className="text-right font-semibold">{value}</span>
    </div>
  );
}

function QualityRow({
  label,
  ok,
  neutral = false,
}: {
  label: string;
  ok: boolean | null | undefined;
  neutral?: boolean;
}) {
  const className = neutral
    ? "border-slate-200 bg-slate-50 text-slate-500"
    : ok
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : "border-amber-200 bg-amber-50 text-amber-700";

  return (
    <div className={`rounded-lg border px-3 py-2 text-xs ${className}`}>
      <div className="flex items-center gap-1.5 font-semibold">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            neutral ? "bg-slate-400" : ok ? "bg-emerald-500" : "bg-amber-500"
          }`}
        />
        {label}
      </div>

      <div className="mt-1 text-[10px] opacity-75">
        {neutral ? "Not required" : ok ? "Available" : "Unavailable"}
      </div>
    </div>
  );
}
