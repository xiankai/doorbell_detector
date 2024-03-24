import fs from 'fs';
import path from 'path';

type Props = {
  audioFiles: {
    filePath: string;
    audioData: string;
  }[];
};

export async function getServerSideProps(): Promise<{ props: Props }> {
  // Fetch file paths here
  const directoryPath = process.env['SAVED_AUDIO_DESTINATION']!;
  const files = fs.readdirSync(directoryPath);
  const audioFiles = files
    .filter((filePath) => filePath.includes('.wav'))
    .map((file) => {
      const filePath = path.join(directoryPath, file);
      return {
        filePath,
        audioData: fs.readFileSync(filePath, { encoding: 'base64' }),
      };
    });

  // Pass file paths as props
  return {
    props: {
      audioFiles,
    },
  };
}

export default function Home({ audioFiles }: Props) {
  const handleAudioFile = (filePath: string, label: string) => async () => {
    const response = await fetch('/api/file', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filePath,
        label,
      }),
    })

    if (response.status === 200) {

    }
  }
  return (
    <>
      <div className='max-w-xl'>
        {/* Show # of files */}
        <table className="table table-xs">
          <thead>
            <tr>
              <th>Audio Filename</th>
              <th>Audio File</th>
              <th>Save as intercom</th>
              <th>Save as background</th>
            </tr>
          </thead>
          <tbody>
            {audioFiles.map(({ filePath, audioData }) => (
              <tr key={filePath}>
                <td>{filePath}</td>
                <td>
                  <audio controls>
                    <source
                      src={`data:audio/mp3;base64,${audioData}`}
                      type="audio/mp3"
                    />
                    Your browser does not support the audio element.
                  </audio>
                </td>
                <td>
                  <button className="btn btn-primary" onClick={handleAudioFile(filePath, 'intercom')}>INTERCOM</button>
                </td>
                <td>
                  <button className="btn btn-primary" onClick={handleAudioFile(filePath, 'background')}>BACKGROUND</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Show player + INTERCOM / BG for each */}
        {/* Automatically upload to EI? */}
        {/* Delete all? */}
      </div>
    </>
  );
}
