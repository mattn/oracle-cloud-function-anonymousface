package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"mime"
	"mime/multipart"
	"strings"

	pigo "github.com/esimov/pigo/core"
	fdk "github.com/fnproject/fdk-go"
	"github.com/nfnt/resize"
	"golang.org/x/image/draw"
)

var (
	maskImg    image.Image
	classifier *pigo.Pigo

	//go:embed data
	fs embed.FS
)

func init() {
	f, err := fs.Open("data/mask.png")
	if err != nil {
		log.Fatal("cannot open mask.png:", err)
	}
	defer f.Close()

	maskImg, _, err = image.Decode(f)
	if err != nil {
		log.Fatal("cannot decode mask.png:", err)
	}

	f, err = fs.Open("data/facefinder")
	if err != nil {
		log.Fatal("cannot open facefinder:", err)
	}
	defer f.Close()

	b, err := ioutil.ReadAll(f)
	if err != nil {
		log.Fatal("cannot read facefinder:", err)
	}

	pigo := pigo.NewPigo()
	classifier, err = pigo.Unpack(b)
	if err != nil {
		log.Fatal("cannot unpack facefinder:", err)
	}
}

type msg struct {
	Message string `json:"message"`
}

func inputErr(out io.Writer, v any) {
	json.NewEncoder(out).Encode(msg{
		Message: fmt.Sprintf("cannot decode input image: %v", v),
	})
}

func main() {
	fdk.Handle(fdk.HandlerFunc(func(ctx context.Context, in io.Reader, out io.Writer) {
		log.Print("start making anonymousface")
		fdk.SetHeader(out, "content-type", "application/json")

		b, err := io.ReadAll(in)
		if err != nil {
			inputErr(out, err)
			return
		}
		fctx, ok := fdk.GetContext(ctx).(fdk.HTTPContext)
		if !ok {
			inputErr(out, "invalid format")
			return
		}
		typ, params, err := mime.ParseMediaType(fctx.ContentType())
		if err != nil {
			inputErr(out, err)
			return
		}

		var img image.Image
		if strings.HasPrefix(typ, "image/") {
			img, _, err = image.Decode(bytes.NewReader(b))
		} else if typ == "multipart/form-data" {
			mr := multipart.NewReader(bytes.NewReader(b), params["boundary"])
			for {
				mp, err := mr.NextPart()
				if err != nil {
					break
				}
				if mp.FormName() == "image" {
					img, _, err = image.Decode(mp)
					if err != nil {
						inputErr(out, err)
						return
					}
					break
				}
			}
			if img == nil {
				inputErr(out, "empty image")
				return
			}
		} else {
		}
		bounds := img.Bounds().Max
		param := pigo.CascadeParams{
			MinSize:     20,
			MaxSize:     2000,
			ShiftFactor: 0.1,
			ScaleFactor: 1.1,
			ImageParams: pigo.ImageParams{
				Pixels: pigo.RgbToGrayscale(pigo.ImgToNRGBA(img)),
				Rows:   bounds.Y,
				Cols:   bounds.X,
				Dim:    bounds.X,
			},
		}
		faces := classifier.RunCascade(param, 0)
		faces = classifier.ClusterDetections(faces, 0.18)

		canvas := image.NewRGBA(img.Bounds())
		draw.Draw(canvas, img.Bounds(), img, image.Point{0, 0}, draw.Over)
		for _, face := range faces {
			pt := image.Point{face.Col - face.Scale/2, face.Row - face.Scale/2}
			fimg := resize.Resize(uint(face.Scale), uint(face.Scale), maskImg, resize.NearestNeighbor)
			log.Println(pt.X, pt.Y, face.Scale)
			draw.Copy(canvas, pt, fimg, fimg.Bounds(), draw.Over, nil)
		}
		fdk.SetHeader(out, "content-type", "image/png")
		err = png.Encode(out, canvas)
		if err != nil {
			log.Print("cannot encode output image:", err)
			fdk.SetHeader(out, "content-type", "application/json")
			json.NewEncoder(out).Encode(msg{
				Message: fmt.Sprintf("cannot encode output image: %v", err),
			})
			return
		}
	}))
}
