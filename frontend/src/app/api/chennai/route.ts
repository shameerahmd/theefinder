export const dynamic = "force-dynamic";

function clamp(
  value: number,
  minimum: number,
  maximum: number,
): number {
  return Math.min(
    maximum,
    Math.max(
      minimum,
      value,
    ),
  );
}


export async function GET(
  request: Request,
) {
  const searchParams =
    new URL(request.url).searchParams;

  const requestedDays = Number(
    searchParams.get("days") ?? "5",
  );

  const requestedMaximum = Number(
    searchParams.get(
      "max_detections",
    ) ?? "25",
  );

  const days = clamp(
    Number.isFinite(requestedDays)
      ? requestedDays
      : 5,
    1,
    5,
  );

  const maxDetections = clamp(
    Number.isFinite(requestedMaximum)
      ? requestedMaximum
      : 25,
    1,
    100,
  );

  const backendBase =
    (
      globalThis as typeof globalThis & {
        process?: {
          env?: {
            THEEFINDER_BACKEND_URL?: string;
          };
        };
      }
    ).process?.env?.THEEFINDER_BACKEND_URL ??
    "http://127.0.0.1:8000";

  const backendUrl =
    `${backendBase}` +
    `/api/classification/chennai` +
    `?days=${days}` +
    `&max_detections=${maxDetections}`;

  try {
    const response = await fetch(
      backendUrl,
      {
        cache: "no-store",
      },
    );

    const text =
      await response.text();

    let data: unknown;

    try {
      data = JSON.parse(text);
    } catch {
      data = {
        detail:
          text ||
          "Backend returned an invalid response.",
      };
    }

    return Response.json(
      data,
      {
        status: response.status,
      },
    );
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unknown backend error.";

    return Response.json(
      {
        application:
          "TheeFinder",

        detail:
          "Unable to connect to the " +
          "TheeFinder FastAPI backend.",

        backend:
          backendUrl,

        error:
          message,
      },
      {
        status: 502,
      },
    );
  }
}