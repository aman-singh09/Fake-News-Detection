const request = require("request")
const _ = require('lodash')
const index = (req, res) => {
    res.send('/api was hit');
}

const pushFormDetails = (req, res, next) => {
    var webBody = {
        newsId: req.body || "The response from web came out Empty"
    }
    var url = 'http://127.0.0.1:5000/api';
    if (_.isEmpty(_.get(webBody, ['newsId', 'title'])) || _.isEmpty(_.get(webBody, ['newsId', 'description']))) {
        res.render('results', {body: "Empty Title or Description"});
        return next();
    }
    var options = {
        method: 'POST',
        body: webBody,
        json: true,
        url: url
    }
    request(options, (error, response, body) => {
        console.log("Making request");
        // console.log(body.data.Authenticity)
        res.render('results', {body: body})
    });
    // res.status(200).json(obj)
}

const webApp = (req, res, next) => {
    res.render('index')
}

var getResultPayload = (req, res, next) => {
    var url = 'http://127.0.0.1:5000/api';
    console.log("Reaching payload api");
    var news = req.query;
    if (_.isEmpty(_.get(news, 'description', ''))) {
        res.json({"result": {error: "Description is Empty", status: 400}})
        return next();
    }
    var requestPayload = {
        newsId: news
    }
    var options = {
        method: 'POST',
        body: requestPayload,
        json: true,
        url: url
    }
    request(options, (error, response, body) => {
        res.status(200).json({"result": body})
    });
}

module.exports = {
    index: index,
    pushFormDetails: pushFormDetails,
    webApp: webApp,
    getResultPayload: getResultPayload
}