import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;

  const rawDays = Number(searchParams.get("days") ?? "5");

  const rawMaxDetections = Number(searchParams.get("max_detections") ?? "25");

  const days = clamp(Number.isFinite(rawDays) ? rawDays : 5, 1, 5);

  const maxDetections = clamp(
    Number.isFinite(rawMaxDetections) ? rawMaxDetections : 25,
    1,
    100,
  );

  const backendBaseUrl =
    process.env.THEEFINDER_BACKEND_URL ?? "http://127.0.0.1:8000";

  const backendUrl =
    `${backendBaseUrl}` +
    `/api/classification/chennai` +
    `?days=${days}` +
    `&max_detections=${maxDetections}`;

  try {
    const response = await fetch(backendUrl, {
      method: "GET",

      cache: "no-store",

      headers: {
        Accept: "application/json",
      },
    });

    const responseText = await response.text();

    let responseData: unknown;

    try {
      responseData = JSON.parse(responseText);
    } catch {
      responseData = {
        detail: responseText || "FastAPI returned an invalid response.",
      };
    }

    return NextResponse.json(responseData, {
      status: response.status,
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unknown backend connection error.";

    return NextResponse.json(
      {
        application: "TheeFinder",

        detail: "Unable to connect to the TheeFinder FastAPI backend.",

        backend_url: backendUrl,

        error: message,
      },
      {
        status: 502,
      },
    );
  }
}
