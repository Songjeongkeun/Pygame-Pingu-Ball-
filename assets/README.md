# Pingu Ball 이미지

두 PNG는 내장 이미지 생성 도구로 준비했으며 실행에 필요한 파일을 프로젝트 안에 보관한다.

- `antarctica.png`: 2:1 남극 배경. 플랫폼이나 장애물이 없는 별도 배경으로 생성했다.
- `pingu.png`: 사용자가 제공한 `디자인_예시/핑구_공_원본.png`를 참조해 검은 사각 배경을 제거한 투명 이미지이다. 얼굴·눈·빨간 부리와 질감을 유지하도록 요청했다.

## 남극 배경 프롬프트

Use case: stylized-concept. Asset type: background image for a Pygame Antarctic platformer called Pingu Ball. Create one panoramic 2:1 landscape scene, target2400x1200. Soft handmade clay-animation aesthetic. Pale powder-blue sky with a few soft clouds, distant Antarctic snow mountains and gentle blue icebergs, calm frozen sea below, very restrained drifting snow. The scene must have low contrast in the center and upper-middle so small foreground game tiles and fish collectibles remain clear. Horizon around lower-middle. No foreground solid ground, no platforms, no blocks, no character, no fish, no igloo, no UI, no grid, no text, no border, no watermark. This is ONLY a quiet beautiful continuous Antarctic backdrop behind a playable level. Soft ambient daylight, icy blue and white palette, inviting and polished.

## 핑구 투명 이미지 프롬프트

Use case: background-extraction. Asset type: Pingu Ball player sprite. Remove the black rectangular background of the supplied Pingu face reference, producing a genuinely transparent PNG with alpha. Preserve the exact original black round clay penguin face, broad red beak, white oval eyes and material texture. Head only, no body or feet. Center the complete isolated head, filling85% of a square canvas with a little transparent padding. Keep its recognizable original proportions and identity. No new scenery, no ground, no cast shadow outside the head, no text, no glow. True transparent background, not a black/white/checkerboard background.
