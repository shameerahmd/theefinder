import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type HistoryDetection = Record<string, unknown>;

interface HistoryResponse {
  detections?: HistoryDetection[];
  [key: string]: unknown;
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;

  const days = searchParams.get("days") ?? "30";
  const classification = searchParams.get("classification");
  const west = searchParams.get("west");
  const south = searchParams.get("south");
  const east = searchParams.get("east");
  const north = searchParams.get("north");
  const limit = searchParams.get("limit") ?? "200";

  const backendUrl =
    process.env.THEEFINDER_BACKEND_URL ??
    "http://127.0.0.1:8000";

  const params = new URLSearchParams();

  params.set("days", days);
  params.set("limit", limit);

  if (classification) {
    params.set("classification", classification);
  }

  if (west) params.set("west", west);
  if (south) params.set("south", south);
  if (east) params.set("east", east);
  if (north) params.set("north", north);

  try {
    const response = await fetch(
      `${backendUrl}/api/classification/history?${params.toString()}`,
      {
        cache: "no-store",
      },
    );

    const data =
      (await response.json()) as HistoryResponse;

    if (Array.isArray(data.detections)) {
      data.detections = data.detections.map(
        (detection) => ({
          ...detection,

          key_features: {
            frp:
              detection.frp,

            distance_to_industry_m:
              detection.distance_to_industry_m,

            industrial_feature_count_5km:
              detection.industrial_feature_count_5km,

            industrial_proximity_score:
              detection.industrial_proximity_score,

            tree_cover_pct:
              detection.tree_cover_pct,

            cropland_pct:
              detection.cropland_pct,

            built_up_pct:
              detection.built_up_pct,

            detections_30d:
              detection.detections_30d,

            active_days_30d:
              detection.active_days_30d,

            persistence_score:
              detection.persistence_score,

            nearest_major_road_distance_m:
              detection.nearest_major_road_distance_m,

            nearest_major_road_class:
              detection.nearest_major_road_class,

            nearest_major_road_name:
              detection.nearest_major_road_name,
          },
        }),
      );
    }

    return NextResponse.json(
      data,
      {
        status: response.status,
      },
    );
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unknown history API error.";

    return NextResponse.json(
      {
        application: "TheeFinder",
        detail:
          "Unable to connect to the TheeFinder history API.",
        error: message,
      },
      {
        status: 502,
      },
    );
  }
}