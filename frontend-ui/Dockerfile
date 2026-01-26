# Build Stage
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production Stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Cloud Run dynamic PORT env var support
CMD sh -c "sed -i 's/listen  .*/listen $PORT;/' /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"
