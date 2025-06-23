Here we provide the VTM-19.1 implementations.
Also, we provide some intermediate results for future investigations.
Implement using compressai.

def run_impl(self, in_filepath, quality, mask_path):
        if not 0 <= quality <= 63:
            raise ValueError(f"Invalid quality value: {quality} (0,63)")

        # Taking 10bit input for now
        bitdepth = 10

        # Convert input image to yuv 444 file
        arr = np.asarray(self._load_img(in_filepath))
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"

        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2**8 - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2**bitdepth - 1)).astype(np.uint16)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=10",
            "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 10]
        if self.rgb:
            cmd.append("--OutputInternalColourSpace=GBRtoRGB")

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint16)
        rec_arr = rec_arr.reshape(arr.shape)

        arr = arr.astype(np.float32) / (2**bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)
        if not self.rgb:
            arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()
        
        _,_, target_mask = get_mask_h_w(mask_path)#(1, 3, H, W)
        total_pixel = target_mask.sum()
        
        bpp = filesize(out_filepath) * 8.0 / (height * width)
      
        bpp_num = filesize(out_filepath) * 8.0
      
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        out = {
            "bpp": bpp,
            "bpp_num": bpp_num,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )

        return out, rec

CLIC results:
rate_vtm="1.1499333509788785, 0.9101794719192918, 0.8102158833272692, 0.7230163026676674, 0.6438561860138645, 0.5718634386560235, 0.5090755973289931, 0.4506051616784891, 0.3978765159446399, 0.3517462894815592, 0.3093580767862514, 0.2709400872654686, 0.23722059408111043, 0.2066153704677026, 0.1544819264960838, 0.11365792020009716, 0.0826026114710305, 0.05923728501034798, 0.0421625593915739"
psnr_vtm="40.69745095306612, 39.60359788726357, 39.06755518826995, 38.56575102508451, 38.04846671744516, 37.52460449940326, 37.02169062728707, 36.50266200918013, 35.98009012119542, 35.47490351172511, 34.96025571627803, 34.435203378291185, 33.93000271129698, 33.41428940663178, 32.34331356608061, 31.31974138286884, 30.332411967237565, 29.367591146061258, 28.423369684694237"

Kodak results:
rate_vtm="1.7304687499999998,1.4329316880967884,1.1870591905381942,1.074011908637153,0.8770005967881946,0.4920179578993056, 0.2870949639214409,0.11255900065104169,0.08086649576822917,0.05742390950520832,0.028895060221354168"
PSNR_vtm="41.458684031865616,40.31617294532954,39.18067591323567,38.578730294969496,37.39594198916618,34.281996204442166,31.839271406495076,28.50603102294186,27.539792095527574,26.607931435014223,24.88610624217208 "
