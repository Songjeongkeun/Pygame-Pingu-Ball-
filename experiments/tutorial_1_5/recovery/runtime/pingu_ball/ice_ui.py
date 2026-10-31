"""준비된 그림을 캐시하고, 그림과 일치하는 논리 화면의 입력 영역을 관리한다."""
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
        self.regular_play_buttons=[self.list_rect('regular_menu',(x+27,y+125,200,53))
                                   for x,y,w,h in self.regular_native_cards]
        self.list_back={name:self.list_rect(name,rect) for name,rect in {
            'tutorial_menu':(1382,28,310,90),'regular_menu':(1382,57,310,92)}.items()}
        self.list_sound={name:pygame.Rect(1340,744,240,78)
                         for name in ('tutorial_menu','regular_menu')}
        self.sound_labels={name:pygame.Rect(1439,763,114,40)
                           for name in self.list_sound}
        self.sound_labels['menu']=pygame.Rect(1439,746,114,40)
        self.sound_labels['title']=pygame.Rect(1419,729,114,40)
        self.sound_faces={}
        for name,rect in (
            ('title',(1320,710,240,78)),
            ('menu',(1340,724,240,78)),
        ):
            face=pygame.image.load(str(root/'assets/ui'/f'{name}_sound_face.png')).convert_alpha()
            target=pygame.Rect(rect)
            if name=='menu':target.move_ip(0,1)
            self.sound_faces[name]=(pygame.transform.smoothscale(face,target.size),target)
        # 두 스테이지 목록에서 동일한 그림·크기·기준선·입력 영역을 사용한다.
        for name,rect in self.list_sound.items():
            self.sound_faces[name]=(self.sound_faces['title'][0],rect)
        source=self.images['regular_menu']
        button=source.subsurface(self.list_rect('regular_menu',(91,345,222,58))).copy()
        # 원본 카드의 사각 조각 없이 둥근 금색 버튼 그림을 유지한다.
        # 크기를 한 번 조절하기 전에 윤곽 가장자리를 매끄럽게 처리한다.
        mask=pygame.Surface((button.get_width()*4,button.get_height()*4),pygame.SRCALPHA)
        pygame.draw.rect(mask,(255,255,255),mask.get_rect(),border_radius=mask.get_height()//2)
        mask=pygame.transform.smoothscale(mask,button.get_size())
        button.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
        self.list_play_button=pygame.transform.smoothscale(button,self.list_rect('regular_menu',(0,0,200,53)).size)
        # 카드 자체의 빈 얼음 무늬로 그림에 박혀 있는 잠금 안내 문구를 가린다.
        # 가장자리를 부드럽게 섞어 열린 카드에 평평한 사각 패널이 생기지 않게 한다.
        blank=source.subsurface(self.list_rect('regular_menu',(500,344,150,18))).copy()
        size=self.list_rect('regular_menu',(0,0,356,49)).size
        face=pygame.transform.smoothscale(blank,size)
        mask=pygame.Surface(size,pygame.SRCALPHA)
        for inset in range(4):
            pygame.draw.rect(mask,(255,255,255,min(255,64*(inset+1))),
                             mask.get_rect().inflate(-2*inset,-2*inset),border_radius=8)
        face.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
        self.regular_unlocked_footer=face
        # 요청받은 그림 수정 영역 이외에는 제공된 화면을 그대로 유지한다.
        for name,filename,rect in (
            ('menu','menu_tutorial_count_removed',(603,586,183,43)),
            ('menu','menu_regular_count_removed',(1088,585,181,44)),
            ('regular_menu','regular_first_button_removed',(83,339,241,70)),
            ('regular_menu','regular_counter_blank',(741,125,235,48)),
            ('tutorial_menu','tutorial_menu_sound_cleared',(1504,812,225,92)),
            ('regular_menu','regular_menu_sound_cleared',(1455,816,275,87)),
        ):
            patch=pygame.image.load(str(root/'assets/ui'/f'{filename}.png')).convert_alpha()
            target=pygame.Rect(rect) if name=='menu' else self.list_rect(name,rect)
            self.images[name].blit(pygame.transform.smoothscale(patch,target.size),target)

    def native_rect(self,name,rect,original_size):
        im=self.images[name];x,y=self.positions[name];sx=im.get_width()/original_size[0];sy=im.get_height()/original_size[1]
        a,b,w,h=rect
        return pygame.Rect(round(x+a*sx),round(y+b*sy),round(w*sx),round(h*sy))

    def blit(self,canvas,name):
        canvas.blit(self.images[name],self.positions[name])

    def list_rect(self,name,rect):
        return self.native_rect(name,rect,(1737,906))

    def next_label_patch(self):
        # 그림에 있는 다음 스테이지 문구를 바꿔야 할 때만 사용한다.
        return self.native_rect('result',(390,950,395,112),(1199,1312))
