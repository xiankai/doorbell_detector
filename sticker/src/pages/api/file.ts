// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";

import fs from 'fs-extra'
import path from 'path'

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const destinationPath = path.join(process.env['SAVED_AUDIO_DESTINATION']!, req.body.label, path.basename(req.body.filePath));
  fs.moveSync(req.body.filePath, destinationPath);

  res.status(200).end()
}
