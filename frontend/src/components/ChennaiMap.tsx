"use client";

import { useEffect, useRef } from "react";
import type { CircleMarker, LayerGroup, Map as LeafletMap } from "leaflet";

export interface MapDetection {
  latitude: number;
  longitude: number;
  frp: number;

  stage_a_prediction: string;
  stage_a_confidence: number;

  stage_b_prediction: string | null;
  stage_b_confidence: number | null;

  final_classification: string;

  acquisition_utc: string;

  data_quality: {
    industrial_context_ok: boolean;
    landcover_ok: boolean;

    persistence_status?: string;
    persistence_ok?: boolean | null;

    road_context_status?: string;
    road_context_ok?: boolean | null;
  };
}

interface ChennaiMapProps {
  detections: MapDetection[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

function getMarkerColor(classification: string): string {
  switch (classification) {
    case "INDUSTRIAL_THERMAL_ANOMALY":
      return "#dc2626";

    case "POTENTIAL_VEHICLE_FIRE":
      return "#f97316";

    case "PERSISTENT_INDUSTRIAL_SOURCE":
      return "#0ea5e9";

    case "FOREST_FIRE":
      return "#16a34a";

    case "AGRICULTURAL_OPEN_BURN":
      return "#d97706";

    case "UNCERTAIN_INDUSTRIAL_EVENT":
      return "#8b5cf6";

    default:
      return "#475569";
  }
}

function formatConfidence(value: number | null): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatClassification(value: string | null | undefined): string {
  if (!value) {
    return "\u2014";
  }

  if (value === "UNCERTAIN_INDUSTRIAL_EVENT") {
    return "Needs Review";
  }

  return value
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export default function ChennaiMap({
  detections,
  selectedIndex,
  onSelect,
}: ChennaiMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const layerRef = useRef<LayerGroup | null>(null);
  const markerRefs = useRef<CircleMarker[]>([]);

  useEffect(() => {
    let disposed = false;

    async function initializeMap() {
      if (containerRef.current === null || mapRef.current !== null) {
        return;
      }

      const L = await import("leaflet");

      if (disposed || containerRef.current === null) {
        return;
      }

      const map = L.map(containerRef.current, {
        center: [12.9716, 80.2707],
        zoom: 9,
        minZoom: 6,
        maxZoom: 18,
      });

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
      }).addTo(map);

      const layer = L.layerGroup().addTo(map);

      mapRef.current = map;
      layerRef.current = layer;

      window.setTimeout(() => {
        map.invalidateSize();
      }, 100);
    }

    void initializeMap();

    return () => {
      disposed = true;
      markerRefs.current = [];

      if (mapRef.current !== null) {
        mapRef.current.remove();
        mapRef.current = null;
      }

      layerRef.current = null;
    };
  }, []);

  useEffect(() => {
    let disposed = false;

    async function drawMarkers() {
      const L = await import("leaflet");

      if (disposed) {
        return;
      }

      const map = mapRef.current;
      const layer = layerRef.current;

      if (map === null || layer === null) {
        window.setTimeout(() => {
          if (!disposed) {
            void drawMarkers();
          }
        }, 100);

        return;
      }

      layer.clearLayers();
      markerRefs.current = [];

      if (detections.length === 0) {
        map.setView([12.9716, 80.2707], 9);
        return;
      }

      const bounds = L.latLngBounds([]);

      detections.forEach((detection, index) => {
        const selected = selectedIndex === index;

        const degraded =
          !detection.data_quality.industrial_context_ok ||
          !detection.data_quality.landcover_ok;

        const markerColor = getMarkerColor(detection.final_classification);

        const marker = L.circleMarker(
          [detection.latitude, detection.longitude],
          {
            radius: selected ? 12 : 8.5,
            color: degraded ? "#111827" : "#ffffff",
            fillColor: markerColor,
            fillOpacity: selected ? 1 : 0.88,
            weight: selected ? 4 : 2,
          },
        );

        const stageB = detection.stage_b_prediction
          ? `
              <div style="display:flex;justify-content:space-between;gap:12px;margin-top:4px;">
                <span style="color:#64748b;">Stage B</span>
                <strong>${formatClassification(
                  detection.stage_b_prediction,
                )}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;gap:12px;margin-top:2px;">
                <span style="color:#64748b;">Confidence</span>
                <strong>${formatConfidence(
                  detection.stage_b_confidence,
                )}</strong>
              </div>
            `
          : "";

        marker.bindPopup(`
          <div style="min-width:220px;font-family:Arial,sans-serif;">
            <div style="font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748b;">
              TheeFinder
            </div>

            <div style="font-size:14px;font-weight:800;margin:4px 0 9px;color:#0f172a;">
              ${formatClassification(detection.final_classification)}
            </div>

            <div style="font-size:12px;line-height:1.6;">
              <div style="display:flex;justify-content:space-between;gap:12px;">
                <span style="color:#64748b;">FRP</span>
                <strong>${detection.frp.toFixed(2)} MW</strong>
              </div>

              <div style="display:flex;justify-content:space-between;gap:12px;margin-top:4px;">
                <span style="color:#64748b;">Stage A</span>
                <strong>${formatClassification(
                  detection.stage_a_prediction,
                )}</strong>
              </div>

              <div style="display:flex;justify-content:space-between;gap:12px;margin-top:2px;">
                <span style="color:#64748b;">Confidence</span>
                <strong>${formatConfidence(
                  detection.stage_a_confidence,
                )}</strong>
              </div>

              ${stageB}

              ${
                degraded
                  ? `
                    <div style="margin-top:8px;border-radius:7px;background:#fffbeb;padding:7px;color:#92400e;font-weight:700;">
                      Context data degraded
                    </div>
                  `
                  : ""
              }
            </div>
          </div>
        `);

        marker.on("click", () => {
          onSelect(index);
        });

        marker.addTo(layer);

        markerRefs.current.push(marker);

        bounds.extend([detection.latitude, detection.longitude]);
      });

      if (detections.length === 1) {
        map.setView([detections[0].latitude, detections[0].longitude], 12);
      } else {
        map.fitBounds(bounds, {
          padding: [40, 40],
          maxZoom: 12,
        });
      }

      if (selectedIndex !== null && markerRefs.current[selectedIndex]) {
        window.setTimeout(() => {
          if (!disposed) {
            markerRefs.current[selectedIndex]?.openPopup();
          }
        }, 120);
      }

      window.setTimeout(() => {
        map.invalidateSize();
      }, 50);
    }

    void drawMarkers();

    return () => {
      disposed = true;
    };
  }, [detections, selectedIndex, onSelect]);

  return (
    <div className="relative h-[520px] w-full lg:h-[620px]">
      <div ref={containerRef} className="h-full w-full" />

      {detections.length === 0 && (
        <div className="pointer-events-none absolute inset-0 z-[500] flex items-center justify-center">
          <div className="rounded-xl border border-slate-200 bg-white/95 px-4 py-3 text-sm text-slate-500 shadow-lg backdrop-blur">
            No Chennai thermal detections available.
          </div>
        </div>
      )}
    </div>
  );
}
