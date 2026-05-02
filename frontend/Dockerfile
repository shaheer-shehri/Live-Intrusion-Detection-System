FROM node:20-alpine AS builder

WORKDIR /app

ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE

COPY package*.json ./
COPY tailwind.config.js postcss.config.js tsconfig.json tsconfig.node.json ./
RUN npm ci

COPY src ./src
COPY index.html vite.config.ts ./

RUN npm run build


FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
