FROM node:20-slim
WORKDIR /app
COPY package.json .
RUN npm install --silent
COPY handler.js .
EXPOSE 8080
CMD ["node", "handler.js"]
