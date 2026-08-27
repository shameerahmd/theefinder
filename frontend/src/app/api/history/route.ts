import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

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
      }
    );

    const data = await response.json();

    return NextResponse.json(
      data,
      { status: response.status }
    );
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          "Unable to connect to the TheeFinder history API.",
      },
      { status: 502 }
    );
  }
}
