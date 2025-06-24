import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string; fileName: string } }
) {
  try {
    const { sessionId, fileName } = params;
    
    // Construct the path to the audio file in PodAgent output directory
    const audioPath = join(process.cwd(), 'PodAgent', 'output', 'sessions', sessionId, 'audio', fileName);
    
    // Read the audio file
    const audioBuffer = await readFile(audioPath);
    
    // Determine content type based on file extension
    const contentType = fileName.endsWith('.wav') ? 'audio/wav' : 
                       fileName.endsWith('.mp3') ? 'audio/mpeg' : 
                       'application/octet-stream';
    
    // Return the audio file with appropriate headers
    return new NextResponse(audioBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `inline; filename="${fileName}"`,
        'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
      },
    });
    
  } catch (error) {
    console.error('Error serving audio file:', error);
    return NextResponse.json(
      { error: 'Audio file not found' },
      { status: 404 }
    );
  }
} 