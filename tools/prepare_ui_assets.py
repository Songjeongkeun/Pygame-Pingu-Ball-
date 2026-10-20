"""Extract the approved UI artwork once; Pillow is not a game dependency.

Usage: python tools/prepare_ui_assets.py --source-dir SOURCE_DIRECTORY
SOURCE_DIRECTORY contains menu.png, pause-no-hint.png, result.png and
play-reference.png from the approved UI review.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1]/'assets/ui'
    target.mkdir(parents=True, exist_ok=True)
    def save(image,name):
        destination=target/(name+'.png')
        temporary=destination.with_suffix('.png.tmp')
        image.save(temporary,format='PNG')
        temporary.replace(destination)
    source = args.source_dir
    menu = Image.open(source/'menu.png').convert('RGBA').resize((1600,829),Image.Resampling.LANCZOS)
    # The icon remains artwork; its on/off wording comes from the game state.
    ImageDraw.Draw(menu).rounded_rectangle((1439,748,1553,788),radius=10,fill='#51c5f0')
    save(menu,'menu')
    for name, filename, bounds in [('pause','pause-no-hint.png',(1020,750)),
                                    ('result','result.png',(810,788))]:
        im = Image.open(source/filename).convert('RGBA')
        if name == 'result':
            d=ImageDraw.Draw(im)
            d.ellipse((511,546,689,711),fill='#fff095',outline='#ffca23',width=5)
            d.rounded_rectangle((320,780,890,843),radius=20,fill='#f1fbff')
            d.rounded_rectangle((385,850,820,917),radius=30,fill='#e9fff9',outline='#47ccd1',width=4)
            alternate=im.copy()
            # A dynamic text field follows the button's gold lighting without
            # stretching the diagonal highlights or its white snowy rim.
            blank=Image.new('RGBA',(410,112));bd=ImageDraw.Draw(blank)
            stops=[(255,222,63),(255,189,20),(255,204,34)]
            for y in range(112):
                t=y/111*2;i=min(1,int(t));a,b=stops[i:i+2];u=t-i
                bd.line((0,y,409,y),fill=tuple(round(a[k]*(1-u)+b[k]*u) for k in range(3))+(255,))
            mask=Image.new('L',blank.size,0)
            ImageDraw.Draw(mask).rounded_rectangle((2,2,407,109),radius=12,fill=255)
            blank.putalpha(mask.filter(ImageFilter.GaussianBlur(2)))
            alternate.alpha_composite(blank,(382,950))
            alternate.thumbnail(bounds,Image.Resampling.LANCZOS)
            save(alternate,'result_next')
        im.thumbnail(bounds,Image.Resampling.LANCZOS)
        save(im,name)
    ref = Image.open(source/'play-reference.png').convert('RGBA')
    def element(box, polygon=None, radius=24):
        patch=ref.crop(box);mask=Image.new('L',patch.size,0);d=ImageDraw.Draw(mask)
        if polygon:d.polygon([(x-box[0],y-box[1]) for x,y in polygon],fill=255)
        else:d.rounded_rectangle((1,1,patch.width-2,patch.height-2),radius=radius,fill=255)
        patch.putalpha(mask)
        return patch
    stage=element((18,10,355,155),[(30,125),(24,89),(38,54),(43,26),(77,24),(115,26),(150,16),(193,18),(220,18),(262,21),(290,32),(323,49),(339,76),(347,114),(335,142),(294,151),(61,153)])
    d=ImageDraw.Draw(stage)
    d.rounded_rectangle((118,32,300,66),radius=9,fill='#0b83c5')
    d.rounded_rectangle((89,68,320,125),radius=13,fill='#4fbeeb')
    save(stage.resize((297,128),Image.Resampling.LANCZOS),'hud_stage')
    timer=element((725,24,1009,111),[(730,56),(750,33),(778,28),(793,25),(823,29),(857,31),(953,32),(979,39),(997,51),(1005,69),(994,92),(972,105),(751,107),(737,94)])
    ImageDraw.Draw(timer).rounded_rectangle((108,29,254,69),radius=10,fill='#b9f0fa')
    save(timer.resize((250,77),Image.Resampling.LANCZOS),'hud_timer')
    for name,box in [('pause',(1161,29,1349,130)),('restart',(1355,29,1536,130)),('select',(1542,29,1722,130))]:
        patch=element(box,radius=31)
        save(patch.resize((round(patch.width*.88),round(patch.height*.88)),Image.Resampling.LANCZOS),'hud_'+name)
    print(target)


if __name__=='__main__':main()
