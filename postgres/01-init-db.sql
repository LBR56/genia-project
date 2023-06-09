-- CREATE TYPE
-- DROP TYPE IF EXISTS genre;
-- CREATE TYPE genre AS ENUM (
--     'ADVENTURE',
--     'HORROR',
--     'COMEDY',
--     'ACTION',
--     'SPORTS'
-- );

-- CREATE TABLE
DROP TABLE IF EXISTS video_meta;
CREATE TABLE "video_meta" (
  "kind" varchar,
  "etag" varchar,
  "id" varchar PRIMARY KEY,
  "publishedAt" varchar,
  "channelId" varchar,
  "title" varchar,
  "description" varchar,
  "thumbnails" varchar,
  "channelTitle" varchar,
  "tags" varchar,
  "categoryId" varchar,
  "liveBroadcastContent" varchar,
  "localized" varchar,
  "defaultAudioLanguage" varchar,
  "defaultLanguage" varchar,
  "duration" varchar,
  "dimension" varchar,
  "definition" varchar,
  "caption" varchar,
  "licensedContent" boolean,
  "contentRating" varchar,
  "projection" varchar,
  "viewCount" bigint,
  "likeCount" bigint,
  "favoriteCount" bigint,
  "commentCount" bigint
);

DROP TABLE IF EXISTS transcript;
CREATE TABLE "transcript" (
  "text" varchar,
  "start" float,
  "duration" float,
  "video_id" varchar
);

COMMENT ON COLUMN "video_meta"."kind" IS 'API 리소스 유형';

COMMENT ON COLUMN "video_meta"."etag" IS 'Etag';

COMMENT ON COLUMN "video_meta"."id" IS 'video id';

COMMENT ON COLUMN "video_meta"."publishedAt" IS '동영상 업로드 날짜(ISO 8601)';

COMMENT ON COLUMN "video_meta"."channelId" IS '업로드 체널 id';

COMMENT ON COLUMN "video_meta"."title" IS '동영상 제목';

COMMENT ON COLUMN "video_meta"."description" IS '동영상 설명';

COMMENT ON COLUMN "video_meta"."thumbnails" IS '미리보기 이미지 맵 딕셔너리';

COMMENT ON COLUMN "video_meta"."channelTitle" IS '체널 이름';

COMMENT ON COLUMN "video_meta"."tags" IS '동영상 태그';

COMMENT ON COLUMN "video_meta"."categoryId" IS 'youtube 동영상 카테고리 ID';

COMMENT ON COLUMN "video_meta"."liveBroadcastContent" IS '라이브 영상 유무';

COMMENT ON COLUMN "video_meta"."localized" IS '현지화 결과';

COMMENT ON COLUMN "video_meta"."defaultAudioLanguage" IS '오디오 언어';

COMMENT ON COLUMN "video_meta"."defaultLanguage" IS '자막 언어';

COMMENT ON COLUMN "video_meta"."duration" IS '동영상 길이(PT#M#S)';

COMMENT ON COLUMN "video_meta"."dimension" IS '3D or 2D';

COMMENT ON COLUMN "video_meta"."definition" IS 'hd or sd';

COMMENT ON COLUMN "video_meta"."caption" IS '동영상 캡션 여부';

COMMENT ON COLUMN "video_meta"."licensedContent" IS '라이센스 콘텐츠 표시 여부';

COMMENT ON COLUMN "video_meta"."contentRating" IS '동영상 등급';

COMMENT ON COLUMN "video_meta"."projection" IS '화면';

COMMENT ON COLUMN "video_meta"."viewCount" IS '동영상 조회된 횟수';

COMMENT ON COLUMN "video_meta"."likeCount" IS '동영상 종하요 사용자 수';

COMMENT ON COLUMN "video_meta"."favoriteCount" IS '즐겨찾기 사용자 수';

COMMENT ON COLUMN "video_meta"."commentCount" IS '댓글 수';

ALTER TABLE "transcript" ADD FOREIGN KEY ("video_id") REFERENCES "video_meta" ("id");
