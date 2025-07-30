import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_POD_URL = process.env.RUNPOD_POD_URL;

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  try {
    const { jobId } = params;

    if (!RUNPOD_POD_URL) {
      return NextResponse.json({ error: 'RunPod pod URL not configured' }, { status: 500 });
    }

    const response = await fetch(`${RUNPOD_POD_URL}/status/${jobId}`);

    if (!response.ok) {
      throw new Error(`Pod API error: ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json({ error: 'Failed to check status' }, { status: 500 });
  }
} 