import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_POD_URL = process.env.RUNPOD_POD_URL || "https://7b59ef18fecc.ngrok-free.app";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;

    console.log('RUNPOD_POD_URL:', RUNPOD_POD_URL);

    if (!RUNPOD_POD_URL) {
      return NextResponse.json({ error: 'RunPod pod URL not configured' }, { status: 500 });
    }

    const response = await fetch(`${RUNPOD_POD_URL}/download/${filename}`);

    if (!response.ok) {
      throw new Error(`Pod API error: ${response.status}`);
    }

    const blob = await response.blob();
    
    return new NextResponse(blob, {
      headers: {
        'Content-Type': 'audio/wav',
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    });

  } catch (error) {
    console.error('Download error:', error);
    return NextResponse.json({ error: 'Failed to download file' }, { status: 500 });
  }
} 