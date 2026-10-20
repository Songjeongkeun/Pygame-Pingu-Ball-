"""Prepared, cached artwork and its matching logical-canvas hit regions."""
import pygame


class IceUI:
    def __init__(self, root):
        names=('menu','pause','result','result_next','hud_stage','hud_timer','hud_pause','hud_restart','hud_select')
        self.images={name:pygame.image.load(str(root/'assets/ui'/f'{name}.png')).convert_alpha() for name in names}
        self.positions={'menu':(0,1),'hud_stage':(18,5),'hud_timer':(686,16),
                        'hud_pause':(1078,13),'hud_restart':(1245,13),'hud_select':(1406,13)}
        for name in ('pause','result','result_next'):
            self.positions[name]=self.images[name].get_rect(center=(800,416)).topleft
        self.menu_buttons={'title':pygame.Rect(917,28,212,70),'help':pygame.Rect(1140,28,214,72),
                           'quit':pygame.Rect(1370,28,185,72),'practice':pygame.Rect(501,635,390,72),
                           'adventure':pygame.Rect(983,635,407,72),'sound':pygame.Rect(1340,730,236,68)}
        self.pause_buttons={name:self.native_rect('pause',rect,(1448,1086)) for name,rect in {
            'close':(1235,258,145,143),'continue':(305,615,836,185),
            'restart':(210,814,488,144),'select':(752,814,494,144)}.items()}
        self.result_buttons={name:self.native_rect('result',rect,(1199,1312)) for name,rect in {
            'next':(247,929,711,145),'retry':(183,1090,402,120),'select':(610,1090,410,120)}.items()}
        for name in ('tutorial_menu','regular_menu'):
            image=pygame.image.load(str(root/'assets/ui'/f'{name}.png')).convert_alpha()
            self.images[name]=pygame.transform.smoothscale(image,(1600,832))
            self.positions[name]=(0,0)
        self.tutorial_cards=[self.list_rect('tutorial_menu',(68+(n%4)*402,168+(n//4)*228,394,214))
                             for n in range(10)]
        self.regular_native_cards=[(58+(n%4)*406,239+(n//4)*194,402,185) for n in range(12)]
        self.regular_cards=[self.list_rect('regular_menu',rect) for rect in self.regular_native_cards]
        self.list_back={name:self.list_rect(name,rect) for name,rect in {
            'tutorial_menu':(1382,28,310,90),'regular_menu':(1382,57,310,92)}.items()}
        self.list_sound={name:self.list_rect(name,rect) for name,rect in {
            'tutorial_menu':(1513,823,205,73),'regular_menu':(1467,830,250,69)}.items()}
        self.list_sound_label={name:self.list_rect(name,rect) for name,rect in {
            'tutorial_menu':(1570,843,126,35),'regular_menu':(1534,846,160,35)}.items()}
        source=self.images['regular_menu']
        button=source.subsurface(self.list_rect('regular_menu',(91,345,222,58))).copy()
        self.list_play_button=pygame.transform.smoothscale(button,(151,41))

    def native_rect(self,name,rect,original_size):
        im=self.images[name];x,y=self.positions[name];sx=im.get_width()/original_size[0];sy=im.get_height()/original_size[1]
        a,b,w,h=rect
        return pygame.Rect(round(x+a*sx),round(y+b*sy),round(w*sx),round(h*sy))

    def blit(self,canvas,name):
        canvas.blit(self.images[name],self.positions[name])

    def list_rect(self,name,rect):
        return self.native_rect(name,rect,(1737,906))

    def next_label_patch(self):
        # Use only when the painted "next stage" wording needs to change.
        return self.native_rect('result',(390,950,395,112),(1199,1312))
